from pathlib import Path

import numpy as np

FLOWER_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold",
    "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris", "globe flower", "purple coneflower",
    "peruvian lily", "balloon flower", "giant white arum lily", "fire lily", "pincushion flower",
    "fritillary", "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort",
    "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion",
    "petunia", "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura",
    "geranium", "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum", "spring crocus",
    "bearded iris", "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose",
    "thorn apple", "morning glory", "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen",
    "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea",
    "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
    "blackberry lily",
]


def load_flowers(root="data", split="train"):
    from torchvision.datasets import Flowers102
    if split == "database":
        parts = [load_flowers(root, "train"), load_flowers(root, "val")]
        return [p for part in parts for p in part[0]], np.concatenate([part[1] for part in parts])
    if split == "all":
        parts = [load_flowers(root, s) for s in ("train", "val", "test")]
        return [p for part in parts for p in part[0]], np.concatenate([part[1] for part in parts])
    dataset = Flowers102(root=root, split=split, download=True)
    paths = [Path(p) for p in dataset._image_files]
    labels = np.array(dataset._labels)
    return paths, labels


def load_news(split="train"):
    from sklearn.datasets import fetch_20newsgroups
    news = fetch_20newsgroups(subset=split, remove=("headers", "footers", "quotes"))
    keep = [i for i, text in enumerate(news.data) if text.strip()]
    texts = [news.data[i].strip() for i in keep]
    labels = np.array(news.target)[keep]
    return texts, labels, list(news.target_names)


def flower_prompts():
    return [f"a photo of a {name}, a type of flower." for name in FLOWER_NAMES]
