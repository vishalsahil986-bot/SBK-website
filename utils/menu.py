MENU: dict = {
    "pizza": [
        "Margherita",
        "Pepperoni",
    ],
    "burger": [
        "Zinger Burger",
        "Beef Burger",
    ],
    "drinks": [
        "Coca-Cola",
        "Mango Lassi",
        "Mineral Water",
    ],
    "sides": [
        "Garlic Bread",
        "Coleslaw",
        "French Fries",
    ],
    "desserts": [
        "Chocolate Brownie",
        "Gulab Jamun",
    ],
}


def get_menu_as_text() -> str:
    """
    Returns the menu formatted as a readable string
    for injection into the system prompt.
    """
    lines = ["📋 SADABAHAR RESTAURANT MENU:", ""]
    for category, items in MENU.items():
        lines.append(f"  {category.upper()}:")
        for item in items:
            lines.append(f"    - {item}")
        lines.append("")
    return "\n".join(lines)


def get_all_item_names() -> list[str]:
    """
    Returns a flat list of all menu item names.
    Used for validation — LLM must not recommend items outside this list.
    """
    return [item for items in MENU.values() for item in items]