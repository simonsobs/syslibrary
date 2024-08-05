import numpy as np
import importlib.resources as pkg_resources
import yaml


def _get_power_file(model, ext='dat'):
    """ File path for the named model
    """
    try:
        with pkg_resources.path('syslibrary.data', f'cl_{model}.{ext}') as path:
            return str(path)
    except FileNotFoundError:
        raise ValueError(f'No template for model {model}')


class Systematic:
    """
    Abstract class for residual definition
    """

    def __init__(self, ell: np.ndarray, nu: list[float] = None, requested_cls: list[str] = None):
        self.ell = np.ascontiguousarray(ell, dtype=np.longlong)
        self.freq = nu
        self.requested_cls = requested_cls


class TemplateResidual(Systematic):
    def __init__(self, ell, nu=None, requested_cls=None, file_root_name=None):

        super().__init__(ell, nu, requested_cls)
        self.template = {}

        if file_root_name:
            self.filename = _get_power_file(file_root_name, 'yaml')

            with open(self.filename) as file:
                file_doc = yaml.full_load(file)

            for spec, doc in file_doc.items():
                for f1, doc_f1 in doc.items():
                    for f2, doc_f1_f2 in doc_f1.items():
                        self.template[spec, f1, f2] = np.asarray(doc_f1_f2)[self.ell]

    def get_delta_cl(self) -> dict:
        return self.template
