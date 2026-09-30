"""Knowledge distillation / fusion of vision foundation models into YOLO11-seg."""

from .models import DistillSegModel, FusionSegModel, export_student, make_trainer
from .teachers import TEACHER_SPECS, get_teacher

__all__ = ["DistillSegModel", "FusionSegModel", "TEACHER_SPECS", "export_student", "get_teacher", "make_trainer"]
