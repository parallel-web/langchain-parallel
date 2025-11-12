from __future__ import annotations

import importlib.util
import os
import sys
import traceback


def load_module_with_deps(file: str, loaded_modules: set[str] | None = None) -> bool:
    """Load a module and handle its dependencies."""
    if loaded_modules is None:
        loaded_modules = set()

    if file in loaded_modules:
        return True

    loaded_modules.add(file)

    # Convert file path to module name for proper package context
    if file.startswith("langchain_parallel/"):
        module_name = file.replace("/", ".").replace(".py", "")

        # Load _client first if this module depends on it
        client_file = "langchain_parallel/_client.py"
        if "_client" not in module_name and client_file not in loaded_modules:
            try:
                load_module_with_deps(client_file, loaded_modules)
            except Exception:
                pass  # Continue if _client fails

        # Use importlib.util.spec_from_file_location with proper module name
        spec = importlib.util.spec_from_file_location(module_name, file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            # Add to sys.modules before execution to support relative imports
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return True
        msg = f"Cannot create spec for {file}"
        raise ImportError(msg)
    # For files outside the package, use the old method
    spec = importlib.util.spec_from_file_location("x", file)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True
    msg = f"Cannot create spec for {file}"
    raise ImportError(msg)


if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False

    # Add the current directory to sys.path to allow package imports
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    loaded_modules: set[str] = set()

    for file in files:
        try:
            load_module_with_deps(file, loaded_modules)
        except Exception:
            has_failure = True
            print(file)  # noqa: T201
            traceback.print_exc()
            print()  # noqa: T201

    sys.exit(1 if has_failure else 0)
