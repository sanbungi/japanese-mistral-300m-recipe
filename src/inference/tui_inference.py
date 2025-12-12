import os
import torch
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal, Container
from textual.widgets import Header, Footer, Input, Button, Label, Static, Log, Select
from textual import work
from transformers import AutoModelForCausalLM, AutoTokenizer

class InferenceApp(App):
    """
    レイアウト修正版 + Input updateエラー修正版
    """
    CSS = """
    /* --- 全体の配置 --- */
    Screen {
        layers: base overlay;
    }

    /* --- サイドバー (左側 30%) --- */
    #sidebar {
        dock: left;
        width: 30%;
        height: 100%;
        background: $panel;
        border-right: vkey $accent;
        padding: 1;
    }

    #sidebar Button {
        width: 100%;
        margin-top: 1;
    }

    .config-label {
        padding-top: 1;
        color: $text-muted;
    }

    /* --- メインエリア (残り全ての領域) --- */
    #main_area {
        height: 100%;
        margin-left: 1;
        padding: 1;
        layout: vertical;
    }

    /* ログエリア: 可能な限り縦に広がる */
    #output_log {
        height: 1fr;
        border: solid $accent;
        background: $surface;
        margin-bottom: 1;
        padding: 1;
    }

    /* 入力エリアコンテナ: 下部に配置 */
    #input_area {
        height: auto;
        dock: bottom;
        margin-bottom: 1;
    }

    /* 入力フォーム: 横幅を最大限使う */
    #user_input {
        width: 1fr;
    }

    /* 生成ボタン: 固定幅 */
    #btn_generate {
        width: 16;
        margin-left: 1;
    }
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded_path = ""
        # デフォルトパスの設定
        self.default_root = "/mnt/hdd/train/results/train/mistral_300m"

    def compose(self) -> ComposeResult:
        """UIのレイアウト定義"""
        yield Header()
        yield Footer()

        # --- サイドバー ---
        with Vertical(id="sidebar"):
            yield Label("Model Root Directory:", classes="config-label")
            # デフォルトパスを指定
            yield Input(placeholder=self.default_root, id="model_root_input", value=self.default_root)
            
            yield Button("Scan Checkpoints", id="btn_scan", variant="default")
            
            yield Label("Select Checkpoint:", classes="config-label")
            yield Select([], id="checkpoint_select", prompt="Scan first...", disabled=True)

            yield Label("Device:", classes="config-label")
            yield Select.from_values(["cuda", "cpu"], value=self.device, id="device_select")

            yield Button("Load Selected Model", id="btn_load", variant="primary", disabled=True)
            
            yield Label("Max New Tokens:", classes="config-label")
            yield Input(placeholder="128", value="100", id="max_tokens_input", type="integer")

            yield Label("Temperature:", classes="config-label")
            yield Input(placeholder="1.0", value="0.7", id="temperature_input", type="number")

            yield Static("\nStatus:", classes="config-label")
            yield Label("[red]Not Loaded[/red]", id="status_lbl")

        # --- メインエリア ---
        with Container(id="main_area"):
            yield Label("Conversation / Output")
            
            # ログ表示エリア
            yield Log(id="output_log", highlight=True)
            
            # 入力エリア
            with Horizontal(id="input_area"):
                yield Input(placeholder="Type your prompt here...", id="user_input")
                yield Button("Generate", id="btn_generate", variant="success", disabled=True)

    def on_mount(self) -> None:
        self.title = "LLM Checkpoint Loader"

    # --- イベントハンドラ ---

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_scan":
            self.action_scan_checkpoints()
        elif event.button.id == "btn_load":
            self.action_load_model()
        elif event.button.id == "btn_generate":
            self.action_generate()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "user_input" and self.model is not None:
            self.action_generate()
        elif event.input.id == "model_root_input":
            self.action_scan_checkpoints()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "checkpoint_select":
            load_btn = self.query_one("#btn_load", Button)
            load_btn.disabled = not event.value

    # --- アクション (ロジック) ---

    @work(exclusive=True, thread=True)
    def action_scan_checkpoints(self) -> None:
        root_path = self.query_one("#model_root_input", Input).value
        select_widget = self.query_one("#checkpoint_select", Select)
        status_lbl = self.query_one("#status_lbl", Label)

        if not os.path.exists(root_path):
            self.call_from_thread(self.notify, f"Path not found: {root_path}", severity="error")
            self.call_from_thread(status_lbl.update, "[red]Path Not Found[/red]")
            return

        try:
            self.call_from_thread(self.notify, f"Scanning {root_path}...")
            
            dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
            
            checkpoints = []
            checkpoints.append(("root (final model)", root_path))

            ckpt_dirs = [d for d in dirs if d.startswith("checkpoint-")]
            
            def get_step(name):
                try:
                    return int(name.split("-")[-1])
                except ValueError:
                    return 0
            
            ckpt_dirs.sort(key=get_step, reverse=True)

            for d in ckpt_dirs:
                full_path = os.path.join(root_path, d)
                checkpoints.append((d, full_path))

            if not checkpoints:
                self.call_from_thread(self.notify, "No checkpoints found.", severity="warning")
            else:
                self.call_from_thread(self.notify, f"Found {len(checkpoints)} options.")

            self.call_from_thread(select_widget.set_options, checkpoints)
            self.call_from_thread(lambda: setattr(select_widget, "disabled", False))
            self.call_from_thread(lambda: setattr(select_widget, "value", checkpoints[0][1]))

        except Exception as e:
            self.call_from_thread(self.notify, f"Scan Error: {str(e)}", severity="error")

    @work(exclusive=True, thread=True)
    def action_load_model(self) -> None:
        path = self.query_one("#checkpoint_select", Select).value
        device = self.query_one("#device_select", Select).value
        status_lbl = self.query_one("#status_lbl", Label)
        
        if not path:
            self.call_from_thread(self.notify, "Please select a checkpoint first.", severity="warning")
            return

        self.call_from_thread(status_lbl.update, "[yellow]Loading...[/yellow]")
        
        try:
            display_name = os.path.basename(path)
            self.call_from_thread(self.notify, f"Loading {display_name} on {device}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForCausalLM.from_pretrained(
                path, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.to(device)
            self.device = device
            self.loaded_path = path
            
            self.call_from_thread(status_lbl.update, f"[green]Loaded: {display_name}[/green]")
            self.call_from_thread(self.notify, "Model loaded successfully!")
            
            self.call_from_thread(lambda: setattr(self.query_one("#btn_generate", Button), "disabled", False))
            self.call_from_thread(self.query_one("#user_input", Input).focus)

        except Exception as e:
            self.call_from_thread(status_lbl.update, "[red]Load Error[/red]")
            self.call_from_thread(self.notify, f"Error: {str(e)}", severity="error")

    @work(exclusive=True, thread=True)
    def action_generate(self) -> None:
        user_input = self.query_one("#user_input", Input)
        prompt = user_input.value
        log_view = self.query_one("#output_log", Log)
        
        if not prompt:
            return

        try:
            max_tokens_val = self.query_one("#max_tokens_input", Input).value
            temperature_val = self.query_one("#temperature_input", Input).value
            max_tokens = int(max_tokens_val) if max_tokens_val else 100
            temperature = float(temperature_val) if temperature_val else 0.7
        except ValueError:
            self.call_from_thread(self.notify, "Invalid parameters", severity="error")
            return

        # 修正: Inputは update() ではなく、valueプロパティを書き換える
        self.call_from_thread(lambda: setattr(user_input, "value", ""))
        
        self.call_from_thread(log_view.write, f"\n[bold cyan]User:[/bold cyan] {prompt}")
        self.call_from_thread(self.notify, "Generating...")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.call_from_thread(log_view.write, f"[bold green]Model ({os.path.basename(self.loaded_path)}):[/bold green] {generated_text}")
            self.call_from_thread(log_view.write, "-" * 40)

        except Exception as e:
            self.call_from_thread(self.notify, f"Generation Error: {str(e)}", severity="error")

if __name__ == "__main__":
    app = InferenceApp()
    app.run()