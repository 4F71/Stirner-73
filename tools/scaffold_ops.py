"""Workspace template generator — `free scaffold <name> --type <type>`."""

import os

from tools.file_ops import WORKSPACE_ROOT

TEMPLATES: dict[str, dict[str, str]] = {
    "torch-experiment": {
        "config.py": '''\
"""Deney ayarlari."""
from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "gpt2"
    batch_size: int = 8
    lr: float = 3e-4
    epochs: int = 3
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "checkpoints"
''',
        "train.py": '''\
"""Egitim dongusu."""
import torch
from config import Config


def train(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    # TODO: DataLoader, model, optimizer buraya

    for epoch in range(cfg.epochs):
        # TODO: egitim adimi
        print(f"Epoch {epoch + 1}/{cfg.epochs}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    # TODO: torch.save(model.state_dict(), ...)


if __name__ == "__main__":
    import os
    train(Config())
''',
        "eval.py": '''\
"""Degerlendirme / inference."""
import torch
from config import Config


def evaluate(cfg: Config, checkpoint_path: str) -> None:
    # TODO: model yukle
    state = torch.load(checkpoint_path, map_location=cfg.device, weights_only=True)
    print(f"Checkpoint yuklendi: {list(state.keys())[:5]} ...")


if __name__ == "__main__":
    import sys
    cfg = Config()
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/model.pt"
    evaluate(cfg, ckpt)
''',
        "requirements.txt": "torch\ntransformers\ndatasets\n",
    },
    "research": {
        "notes.md": "# Araştırma Notları\n\n## Amaç\n\n## Kaynaklar\n\n## Bulgular\n",
        "scratchpad.py": '''\
"""Hizli prototipleme ve veri kesifi."""
# Buraya her seyi yaz, temizlemek zorunda degilsin.
''',
        "summary.md": "# Özet\n\n## TL;DR\n\n## Detaylar\n",
    },
    "generic": {
        "main.py": '''\
"""Giris noktasi."""


def main() -> None:
    pass


if __name__ == "__main__":
    main()
''',
        "utils.py": '"""Yardimci fonksiyonlar."""\n',
        "README.md": "# Proje\n\n## Kullanim\n\n```bash\npython main.py\n```\n",
    },
}

AVAILABLE_TYPES = list(TEMPLATES.keys())

# Her sablon icin uretim sonrasi onerilen ilk komut/adim.
NEXT_STEPS: dict[str, str] = {
    "torch-experiment": "cd workspace/{name} && pip install -r requirements.txt && python train.py",
    "research": "cd workspace/{name} && python scratchpad.py  (notlar: notes.md)",
    "generic": "cd workspace/{name} && python main.py",
}


def scaffold(name: str, template_type: str = "generic") -> str:
    if template_type not in TEMPLATES:
        return (
            f"Bilinmeyen sablon tipi: '{template_type}'. "
            f"Gecerli tipler: {', '.join(AVAILABLE_TYPES)}"
        )

    target_dir = os.path.join(WORKSPACE_ROOT, name)
    if os.path.exists(target_dir):
        return f"Hata: '{target_dir}' zaten var. Farkli bir isim sec."

    os.makedirs(target_dir, exist_ok=True)
    files = TEMPLATES[template_type]
    created = []
    for filename, content in files.items():
        path = os.path.join(target_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        created.append(f"  workspace/{name}/{filename}")

    next_step = NEXT_STEPS.get(template_type, "cd workspace/{name}").format(name=name)
    return (
        f"'{template_type}' sablonu olusturuldu: workspace/{name}/\n"
        + "\n".join(created)
        + f"\n\nSonraki adim: {next_step}"
    )
