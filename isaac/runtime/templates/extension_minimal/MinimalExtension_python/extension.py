import omni.ext

__all__ = ["Extension"]


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        pass

    def on_shutdown(self) -> None:
        pass
