from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class TemplateItem:
    temperature: float
    image: Any
    x: float
    y: float
    z: float

@dataclass
class CorrectionItem:
    temperature: float
    image: Any
    x: float
    y: float
    z: float

class RotationCorrection:
    def __init__(self):
        self.template: Dict[str, Dict[str, TemplateItem]] = {}  # name -> item_id -> TemplateItem
        self.correction_list: Dict[str, Dict[float, Dict[str, CorrectionItem]]] = {}  # name -> temp -> item_id -> CorrectionItem

    def add_template_item(self, name: str, item_id: str, item: TemplateItem):
        if name not in self.template:
            self.template[name] = {}
        self.template[name][item_id] = item

    def add_correction_item(self, name: str, temperature: float, item_id: str, item: CorrectionItem):
        if name not in self.correction_list:
            self.correction_list[name] = {}
        if temperature not in self.correction_list[name]:
            self.correction_list[name][temperature] = {}
        self.correction_list[name][temperature][item_id] = item

    def get_template_item(self, name: str, item_id: str) -> Optional[TemplateItem]:
        return self.template.get(name, {}).get(item_id)

    def get_correction_item(self, name: str, temperature: float, item_id: str) -> Optional[CorrectionItem]:
        return self.correction_list.get(name, {}).get(temperature, {}).get(item_id)

# --- Test Code ---

rc = RotationCorrection()

rc.add_template_item("DUT_1", "small", TemplateItem(22, "image1", 1.0, 2.0, 3.0))
rc.add_template_item("DUT_1", "large", TemplateItem(22, "image1", 1.0, 2.0, 3.0))
rc.add_template_item("DUT_2", "small", TemplateItem(22, "image2", 1.0, 2.0, 3.0))
rc.add_template_item("DUT_2", "large", TemplateItem(22, "image2", 1.0, 2.0, 3.0))

rc.add_correction_item("DUT_1", 50.0, "small", CorrectionItem(50.0, "imgA", 0, 0, 0))
rc.add_correction_item("DUT_1", 50.0, "large", CorrectionItem(50.0, "imgA", 0, 0, 0))
rc.add_correction_item("DUT_1", 80.0, "small", CorrectionItem(80.0, "imgA", 0, 0, 0))
rc.add_correction_item("DUT_1", 80.0, "large", CorrectionItem(80.0, "imgA", 0, 0, 0))
rc.add_correction_item("DUT_2", 50.0, "small", CorrectionItem(50.0, "imgA", 0, 0, 0))
rc.add_correction_item("DUT_2", 50.0, "large", CorrectionItem(50.0, "imgA", 0, 0, 0))
rc.add_correction_item("DUT_2", 80.0, "small", CorrectionItem(80.0, "imgA", 0, 0, 0))
rc.add_correction_item("DUT_2", 80.0, "large", CorrectionItem(80.0, "imgA", 0, 0, 0))

template = rc.get_template_item("DUT_2", "large")
correction = rc.get_correction_item("DUT_2",80,"small")


print(rc)

# You can now print or debug individual items