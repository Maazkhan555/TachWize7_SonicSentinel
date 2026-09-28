"""
model_service.py — SonicSentinel CNN model loading and inference.

Architecture (exact match for sonicsentinel_fsd50k_193class_best.pt):
  ConvBlock: Conv2d → BN → ReLU → MaxPool2d
  Blocks: 1→32, 32→64, 64→128, 128→256
  AdaptiveAvgPool2d → Flatten → Linear(256,256) → ReLU → Dropout → Linear(256,193)

Loaded ONCE at startup. Thread-safe for inference.
"""

import os
import time
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── FSD50K 193-class label list ───────────────────────────────────────────────
# Derived from the training manifest target_class column (sorted alphabetically
# to match the order used during training — index = class index in checkpoint).
FSD50K_CLASSES = [
    "Acoustic_guitar", "Aircraft", "Alarm", "Animal", "Applause",
    "Bark", "Bass_drum", "Bass_guitar", "Bathtub_(filling_or_washing)",
    "Bell", "Bicycle", "Bicycle_bell", "Bird",
    "Bird_vocalization_and_bird_call_and_bird_song", "Boat_and_Water_vehicle",
    "Boiling", "Boom", "Bowed_string_instrument", "Brass_instrument",
    "Breathing", "Burping_and_eructation", "Bus", "Buzz",
    "Camera", "Car", "Car_passing_by", "Cat", "Chatter",
    "Cheering", "Chewing_and_mastication", "Chime", "Chink_and_clink",
    "Chirp_and_tweet", "Church_bell", "Clapping", "Clock",
    "Coin_(dropping)", "Conversation", "Cough", "Cowbell", "Crack",
    "Crackle", "Crash_cymbal", "Cricket", "Crow",
    "Crowd", "Crushing", "Crumpling_and_crinkling",
    "Crying_and_sobbing", "Cutlery_and_silverware", "Cymbal",
    "Dishes_and_pots_and_pans", "Dog", "Door",
    "Domestic_animals_and_pets", "Domestic_sounds_and_home_sounds",
    "Doorbell", "Drawer_open_or_close", "Drill", "Drip",
    "Drum", "Drum_kit", "Electric_guitar", "Engine",
    "Engine_starting", "Explosion", "Fart", "Female_singing",
    "Female_speech_and_woman_speaking", "Fill_(with_liquid)",
    "Finger_snapping", "Fire", "Fireworks",
    "Fixed-wing_aircraft_and_airplane", "Fowl", "Frying_(food)",
    "Gasp", "Giggle", "Glass", "Glockenspiel", "Gong",
    "Growling", "Guitar", "Gunshot_and_gunfire", "Gurgling",
    "Hammer", "Hands", "Harmonica", "Harp", "Hi-hat", "Hiss",
    "Human_group_actions", "Human_voice", "Insect",
    "Keys_jangling", "Keyboard_(musical)", "Knock",
    "Laughter", "Liquid", "Livestock_and_farm_animals_and_working_animals",
    "Male_speech_and_man_speaking", "Mechanisms", "Meow",
    "Microwave_oven", "Motor_vehicle_(road)", "Motorcycle",
    "Musical_instrument", "Music", "Ocean", "Organ",
    "Percussion", "Piano", "Plucked_string_instrument",
    "Power_tool", "Printer", "Purr", "Rail_transport",
    "Rain", "Raindrop", "Ratchet_and_pawl", "Rattle",
    "Rattle_(instrument)", "Respiratory_sounds", "Ringtone", "Run",
    "Sawing", "Scissors", "Screaming", "Screech",
    "Scratching_(performance_technique)", "Shatter", "Shout",
    "Singing", "Sink_(filling_or_washing)", "Siren",
    "Skateboard", "Slam", "Sliding_door", "Sneeze",
    "Snare_drum", "Speech", "Splash_and_splatter", "Squeak",
    "Stream", "Subway_and_metro_and_underground", "Tabla",
    "Tambourine", "Tap", "Tearing", "Telephone",
    "Thump_and_thud", "Tick", "Toilet_flush", "Tools",
    "Traffic_noise_and_roadway_noise", "Train", "Truck",
    "Typewriter", "Typing", "Vehicle", "Walk_and_footsteps",
    "Water", "Water_tap_and_faucet", "Waves_and_surf",
    "Whispering", "Whoosh_and_swoosh_and_swish", "Wind",
    "Wind_chime", "Wind_instrument_and_woodwind_instrument",
    "Wood", "Writing", "Zipper_(clothing)",
]

# Pad / trim to exactly 193 entries
while len(FSD50K_CLASSES) < 193:
    FSD50K_CLASSES.append(f"Class_{len(FSD50K_CLASSES)}")
FSD50K_CLASSES = FSD50K_CLASSES[:193]

# ── Classes considered high-risk for dashboard alerting ──────────────────────
HIGH_RISK_CLASSES = {
    "Gunshot_and_gunfire", "Explosion", "Screaming", "Shout",
    "Siren", "Alarm", "Fireworks", "Boom", "Shatter",
    "Crying_and_sobbing", "Growling",
}


# ── Model architecture ────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)


class SonicSentinelCNN(nn.Module):
    def __init__(self, num_classes: int = 193):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
        )
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ── Singleton service ─────────────────────────────────────────────────────────

class ModelService:
    _instance = None

    def __init__(self):
        self.model      = None
        self.device     = None
        self.loaded     = False
        self.error      = None
        self.model_path = None
        self.classes    = FSD50K_CLASSES  # overwritten by _load() if checkpoint has real labels
        self._load()

    @classmethod
    def get(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        base = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base, "SonicSentinel_Persistent 0.1", "models",
                         "sonicsentinel_fsd50k_193class_best.pt"),
            os.path.join(base, "models", "sonicsentinel_fsd50k_193class_best.pt"),
        ]
        pt_path = next((p for p in candidates if os.path.isfile(p)), None)
        if pt_path is None:
            self.error = "Model checkpoint not found."
            logger.error(self.error)
            return

        self.model_path = pt_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("[SS-CNN] Loading on %s: %s", self.device, pt_path)

        try:
            ckpt = torch.load(pt_path, map_location=self.device, weights_only=False)

            # Extract real class labels from checkpoint
            if isinstance(ckpt, dict) and "classes" in ckpt:
                self.classes = list(ckpt["classes"])
                logger.info("[SS-CNN] Loaded %d real class labels from checkpoint", len(self.classes))
            else:
                self.classes = FSD50K_CLASSES  # fallback
                logger.warning("[SS-CNN] No 'classes' key in checkpoint, using fallback list")

            num_classes = len(self.classes)
            model = SonicSentinelCNN(num_classes=num_classes)

            state = ckpt
            if isinstance(ckpt, dict):
                key = next((k for k in ("model_state_dict", "state_dict", "model") if k in ckpt), None)
                state = ckpt[key] if key else ckpt

            model.load_state_dict(state, strict=True)
            model.to(self.device)
            model.eval()
            self.model  = model
            self.loaded = True
            logger.info("[SS-CNN] Loaded successfully. %d classes, device=%s", num_classes, self.device)
        except Exception as e:
            self.error = str(e)
            logger.exception("[SS-CNN] Failed to load: %s", e)

    @torch.inference_mode()
    def predict(self, tensor: torch.Tensor) -> dict:
        """
        Run inference on a preprocessed tensor [1,1,64,T].
        Returns dict with label, confidence, top_predictions.
        """
        if not self.loaded:
            raise RuntimeError(self.error or "Model not loaded.")

        t0 = time.perf_counter()
        tensor = tensor.to(self.device)
        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        ms     = (time.perf_counter() - t0) * 1000

        top_k   = int(min(5, len(probs)))
        top_idx = probs.argsort()[::-1][:top_k]

        top_preds = [
            {"label": self.classes[i], "confidence": round(float(probs[i]), 4)}
            for i in top_idx
        ]
        best = top_preds[0]

        return {
            "label":           best["label"],
            "confidence":      best["confidence"],
            "top_predictions": top_preds,
            "inference_ms":    round(ms, 1),
            "device":          str(self.device),
        }

    def status(self) -> dict:
        return {
            "loaded":      self.loaded,
            "model_name":  "SonicSentinel CNN (FSD50K)",
            "device":      str(self.device) if self.device else "N/A",
            "model_path":  os.path.basename(self.model_path) if self.model_path else "N/A",
            "num_classes": len(self.classes),
            "error":       self.error,
        }
