# Contributing to Climax Heroes AI 🎮

First off, thank you for considering contributing to Climax Heroes AI! It's people like you who make this a fun, viable project for the community.

Here are some guidelines to help you get started.

---

## ✦ Coding Style

Please follow these coding guidelines when submitting contributions:

*   **Snake_case:** Use `snake_case` for all variable and function names.
*   **Flat Logic:** Avoid deep nesting of conditions (`if` statement inside `if` statement). Use early returns to simplify control flow.
*   **Readability:** Keep code readable, clean, and well-spaced. Minimize cognitive load.
*   **Error Handling:** Value clarity and explicit failure. Let the program fail fast if critical dependencies (like the gamepad driver or emulator window capture) are not working.

---

## ✦ Development Workflow

### 1. Prerequisites
Ensure you have the virtual gamepad driver (**ViGEmBus**) installed on your Windows machine, and configured your PCSX2 emulator to **16:9 fullscreen (1920x1080 resolution)**.

### 2. Environment Setup
Create a virtual environment and install the package dependencies:
```powershell
python312 -m venv .venv
.\.venv\Scripts\Activate.ps1
python312 -m pip install -r requirements.txt
```

### 3. Testing Changes
Before committing, make sure to test that all components are functioning:
*   Verify gamepad emulation:
    ```powershell
    python312 .\tools\test_gamepad.py
    ```
*   Verify window capture and coordinate alignments:
    ```powershell
    python312 .\tools\screen_capture_helper.py
    ```
*   Run the custom environment with console dashboard logging:
    ```powershell
    python312 .\tests\test_env.py
    ```

---

## ✦ How to Contribute

1.  **Fork** the repository and create your feature branch from `master` (e.g., `feature/amazing-feature`).
2.  Commit your changes with clear, descriptive commit messages.
3.  Push your branch to your fork.
4.  Open a **Pull Request** explaining your modifications, the problem you are solving, and how you tested the changes.

Thank you for helping train the ultimate Rider! 🏍️⚡
