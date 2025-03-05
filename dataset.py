from torch.utils.data import Dataset

import json
from pathlib import Path
from typing import Any, TypeAlias


CardDict: TypeAlias = dict[str, Any]


class MTGCardDataset(Dataset):
    def __init__(self, json_file_path: Path) -> None:
        self.card_dicts = read_card_dicts_from_json_path(json_file_path)

    def __len__(self) -> int:
        return len(self.card_dicts)

    def __getitem__(self, i: int) -> str:
        card_dict = self.card_dicts[i]
        return card_dict_to_training_string(card_dict)


def card_dict_to_training_string(card: CardDict) -> str:
    # card['originalText'] is what was originally printed on the card, while card['text'] is the rules text with any
    # updates. For example, cards like Archon of Justice in the EVE set have 'remove from the game' in 'originalText'
    # and 'exile' in 'text'.

    # TODO: add Loyalty field for planeswalkers
    # TODO: should we skip fields like power, toughness, and text when they're not applicable?
    # Some cards have no mana cost, like Ancestral Vision.
    mana_cost = card.get("manaCost", "None")
    text = card.get("text", "None")
    power = card.get("power", "None")
    toughness = card.get("toughness", "None")
    # I removed 'rarity' because the HuggingFace MTG dataset doesn't have it.
    return f"""name: {card["name"]} manaCost: {mana_cost} type: {card["type"]} text: {text} power: {power} toughness: {toughness}"""


def read_card_dicts_from_json_path(json_file_path: Path) -> list[CardDict]:
    with open(str(json_file_path), "r") as f:
        raw_json_data = json.load(f)

    all_card_dicts = []
    for mtg_set_code in raw_json_data['data'].keys():
        all_card_dicts.extend(raw_json_data['data'][mtg_set_code]['cards'])

    return all_card_dicts
