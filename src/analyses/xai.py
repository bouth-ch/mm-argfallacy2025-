import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from transformers import AutoTokenizer
from mamkit.models.text import Transformer

from src.utils.fold_manifest import dialogue_for_fold


_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = _ROOT / 'notebooks' / 'figures'
CHECKPOINTS_DIR = _ROOT / 'checkpoints'

# Special tokens to skip in all plots
_SPECIAL = {'<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]', '<mask>'}


class ModelAnalyzer:
    """
    Loads a fine-tuned checkpoint and runs XAI analysis
    (attention heatmap + gradient saliency) on selected snippets.

    Works for any experiment that uses the Transformer (mamkit) architecture:
    RoBERTa, DeBERTa, etc.
    """

    def __init__(self, experiment_name, config, results_path=None):
        if results_path is None:
            results_path = str(_ROOT / 'results' / 'results.json')
        self.experiment_name = experiment_name
        self.config          = config
        self.task_name       = config.get('task_name', 'afc')
        self.results_path    = results_path
        self.tokenizer       = AutoTokenizer.from_pretrained(
            config['model_card'],
            cache_dir=str(_ROOT / 'data' / 'hf_cache')
        )

    def get_checkpoint_folds(self):
        """Return set of fold indices for which checkpoints exist."""
        ckpt_base = CHECKPOINTS_DIR / self.experiment_name
        if not ckpt_base.exists():
            return set()
        return {
            int(p.name.split('_')[1])
            for p in ckpt_base.iterdir()
            if p.is_dir() and p.name.startswith('fold_')
            and (p / 'best.ckpt').exists()
        }

    def load_model(self, fold_idx):
        """Load weights-only checkpoint for a given fold (0-indexed)."""
        ckpt_path = (CHECKPOINTS_DIR / self.experiment_name
                     / f'fold_{fold_idx}' / 'best.ckpt')
        model = Transformer(
            model_card=self.config['model_card'],
            head=self.config['head'],
            dropout_rate=self.config['dropout_rate'],
            is_transformer_trainable=self.config['is_transformer_trainable']
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckpt = torch.load(str(ckpt_path), map_location=device)
        state_dict = {k[len('model.'):]: v
                      for k, v in ckpt['state_dict'].items()
                      if k.startswith('model.')}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    def get_snippets(self, fold_info, loader):
        """Return all rows for the held-out dialogue (AFC: snippets; AFD: sentences)."""
        fold_idx = int(fold_info['fold_idx'])
        fold_dialogue = dialogue_for_fold(
            self.experiment_name,
            fold_idx,
            results_path=self.results_path,
            strict=True,
        )
        df = loader.data[
            loader.data['dialogue_id'] == fold_dialogue
        ].reset_index(drop=True).copy()
        if self.task_name == 'afc':
            df['label'] = df['fallacy'].astype(int)
            df['model_text'] = df['snippet']
        else:
            df['label'] = df['label'].astype(int)
            df['model_text'] = df['sentence']
        return df

    @staticmethod
    def _enc_to_model_device(enc, model):
        """Move tokenizer outputs to the same device as ``model`` (fixes CUDA vs CPU)."""
        dev = next(model.parameters()).device
        return {
            k: v.to(dev) if torch.is_tensor(v) else v
            for k, v in enc.items()
        }

    @staticmethod
    def _stratified_by_true_label(df, *, max_per_class: int, label_col: str = 'label'):
        """
        Up to ``max_per_class`` rows per distinct gold label (first rows in dataframe order).

        Avoids the bias of ``.head(k)``, which often picks only the majority classes.
        """
        if df.empty or max_per_class <= 0:
            return df.iloc[:0].copy()
        chunks = []
        for cls in sorted(df[label_col].unique()):
            sub = df[df[label_col] == cls].head(max_per_class)
            chunks.append(sub)
        return pd.concat(chunks, ignore_index=True)

    # ------------------------------------------------------------------
    # Token → word merging
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_subwords(tokens, scores):
        """
        Merge BPE/SentencePiece subword tokens to word level.

        RoBERTa uses Ġ prefix for word-start tokens.
        BERT uses ## prefix for continuations.
        Scores are max-pooled across subwords of the same word.
        Special tokens are dropped.
        """
        words, word_scores = [], []
        cur_word, cur_scores = '', []

        for tok, sc in zip(tokens, scores):
            if tok in _SPECIAL:
                continue

            if tok.startswith('Ġ') or tok.startswith('▁'):
                # flush previous word
                if cur_word:
                    words.append(cur_word)
                    word_scores.append(float(np.max(cur_scores)))
                cur_word   = tok[1:]
                cur_scores = [sc]
            elif tok.startswith('##'):
                cur_word  += tok[2:]
                cur_scores.append(sc)
            else:
                # plain token (BERT CLS/SEP already filtered; first token of seq)
                if cur_word:
                    words.append(cur_word)
                    word_scores.append(float(np.max(cur_scores)))
                cur_word   = tok
                cur_scores = [sc]

        if cur_word:
            words.append(cur_word)
            word_scores.append(float(np.max(cur_scores)))

        return words, np.array(word_scores)

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------

    def get_attention(self, model, snippet):
        """Last-layer CLS attention averaged over heads (word-level)."""
        enc = self.tokenizer(snippet, return_tensors='pt',
                             truncation=True, max_length=128, padding=False)
        enc = self._enc_to_model_device(enc, model)
        with torch.no_grad():
            out = model.model(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask'],
                output_attentions=True
            )
        last_attn = out.attentions[-1][0]           # [heads, seq, seq]
        cls_attn  = last_attn[:, 0, :].mean(dim=0)  # avg over heads → [seq]
        tokens    = self.tokenizer.convert_ids_to_tokens(
            enc['input_ids'][0].detach().cpu().tolist()
        )
        words, word_scores = self._merge_subwords(
            tokens, cls_attn.detach().cpu().numpy()
        )
        return words, word_scores

    # ------------------------------------------------------------------
    # Gradient saliency
    # ------------------------------------------------------------------

    def get_saliency(self, model, snippet):
        """Gradient of predicted class score w.r.t. input embeddings (word-level L2 norm)."""
        enc = self.tokenizer(snippet, return_tensors='pt',
                             truncation=True, max_length=128, padding=False)
        enc = self._enc_to_model_device(enc, model)
        input_ids      = enc['input_ids']
        attention_mask = enc['attention_mask']

        embeddings = model.model.embeddings(input_ids)
        embeddings.retain_grad()

        out      = model.model(inputs_embeds=embeddings,
                               attention_mask=attention_mask)
        mask     = attention_mask.float()
        text_emb = (out.last_hidden_state * mask[:, :, None]).sum(dim=1)
        text_emb = model.dropout(text_emb) / mask.sum(dim=1)[:, None]
        logits   = model.head(text_emb)

        pred_class = logits.argmax(dim=-1).item()
        logits[0, pred_class].backward()

        token_saliency = embeddings.grad[0].norm(dim=-1).detach().cpu().numpy()
        tokens         = self.tokenizer.convert_ids_to_tokens(
            input_ids[0].detach().cpu().tolist()
        )
        words, word_scores = self._merge_subwords(tokens, token_saliency)
        return words, word_scores, pred_class

    # ------------------------------------------------------------------
    # SHAP (text partition / permutation explainer)
    # ------------------------------------------------------------------

    def _create_shap_text_explainer(self, model, *, algorithm: str = 'partition'):
        """
        Build a SHAP explainer for string inputs → softmax class probabilities.

        Uses the library's ``Text`` masker (word-ish segments). Requires ``shap``.
        """
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "method='shap' requires the `shap` package. "
                "Install with: pip install 'shap>=0.44'"
            ) from e

        tokenizer = self.tokenizer
        device = next(model.parameters()).device

        def f(texts: list) -> np.ndarray:
            probs = []
            with torch.no_grad():
                for text in texts:
                    enc = tokenizer(
                        text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=128,
                        padding=True,
                    )
                    enc = {
                        k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in enc.items()
                    }
                    out = model.model(
                        input_ids=enc['input_ids'],
                        attention_mask=enc['attention_mask'],
                    )
                    mask = enc['attention_mask'].float()
                    text_emb = (out.last_hidden_state * mask[:, :, None]).sum(dim=1)
                    text_emb = model.dropout(text_emb) / mask.sum(dim=1)[:, None]
                    logits = model.head(text_emb)
                    p = torch.softmax(logits, dim=-1).cpu().numpy()
                    probs.append(p[0])
            return np.stack(probs, axis=0)

        masker = shap.maskers.Text(tokenizer)
        return shap.Explainer(f, masker, algorithm=algorithm)

    def _explain_shap_snippet(
        self,
        explainer,
        model,
        snippet: str,
        *,
        max_evals: int,
    ) -> tuple[list[str], np.ndarray, int]:
        """
        Run SHAP on one string; return word-level segments, |φ| for predicted class, pred index.
        """
        with torch.no_grad():
            te = self._quick_embed(model, snippet)
            pred_class = int(model.head(te).argmax(dim=-1).item())

        sv = explainer([snippet], max_evals=max_evals)
        segs = sv.data[0]
        phi = np.asarray(sv.values[0, :, pred_class], dtype=np.float64)
        phi_abs = np.abs(phi)

        words: list[str] = []
        scores: list[float] = []
        for seg, sc in zip(segs, phi_abs):
            w = seg.strip()
            if not w:
                continue
            words.append(w)
            scores.append(float(sc))

        if not words:
            words = ['<empty>']
            scores = [0.0]
        return words, np.array(scores, dtype=np.float64), pred_class

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_word_heatmap(words, scores, ax, cmap_name='Blues', words_per_row=10):
        """
        Word-level heatmap with line wrapping.
        Each word is a colored box; rows wrap at `words_per_row`.
        """
        scores = np.array(scores)
        norm   = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        cmap   = plt.get_cmap(cmap_name)

        rows       = [words[i:i + words_per_row]
                      for i in range(0, len(words), words_per_row)]
        row_norms  = [norm[i:i + words_per_row]
                      for i in range(0, len(norm), words_per_row)]
        n_rows     = len(rows)

        ax.set_xlim(0, words_per_row)
        ax.set_ylim(0, n_rows)
        ax.set_aspect('auto')

        for row_i, (rw, rs) in enumerate(zip(rows, row_norms)):
            y = n_rows - row_i - 1
            for col_i, (word, s) in enumerate(zip(rw, rs)):
                color = cmap(0.15 + 0.85 * s)
                ax.add_patch(plt.Rectangle(
                    (col_i, y + 0.08), 0.93, 0.84,
                    color=color, linewidth=0
                ))
                fontsize  = 9 if words_per_row <= 12 else 8
                textcolor = 'white' if s > 0.55 else '#1a1a1a'
                ax.text(col_i + 0.465, y + 0.5, word,
                        ha='center', va='center',
                        fontsize=fontsize, color=textcolor, fontweight='bold')
        ax.axis('off')

    @staticmethod
    def _plot_top_words_bar(words, scores, ax, top_n=10, cmap_name='Blues'):
        """Horizontal bar chart of top-N most important words."""
        scores = np.array(scores)
        norm   = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

        top_idx    = np.argsort(norm)[-top_n:][::-1]
        top_words  = [words[i] for i in top_idx]
        top_scores = norm[top_idx]

        cmap   = plt.get_cmap(cmap_name)
        colors = [cmap(0.2 + 0.8 * s) for s in top_scores]

        y_pos = list(range(len(top_words)))
        ax.barh(y_pos[::-1], top_scores, color=colors, edgecolor='none')
        ax.set_yticks(y_pos[::-1])
        ax.set_yticklabels(top_words, fontsize=9)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel('Normalised importance', fontsize=8)
        ax.set_title(f'Top-{top_n} words', fontsize=9, fontweight='bold')
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    # ------------------------------------------------------------------
    # Main plot entry point
    # ------------------------------------------------------------------

    def plot_xai(self, selected_folds, loader, label_names, method='saliency',
                 words_per_row=10, top_n=10,
                 correct_per_true_class: int = 1,
                 wrong_per_true_class: int = 1,
                 fold_roles: set[str] | None = None,
                 shap_algorithm: str = 'partition',
                 shap_max_evals: int = 400):
        """
        For each selected fold: one figure with one row per sample.
        Each row shows:
          LEFT  (70%): word-level heatmap with line wrapping
          RIGHT (30%): top-N important words bar chart

        Parameters
        ----------
        method       : 'saliency' (recommended), 'attention', or 'shap'
        words_per_row: words per line in the heatmap panel
        top_n        : how many words to rank in the bar chart
        correct_per_true_class
            Max **correct** snippets per **gold** class (spread across fallacy types).
        wrong_per_true_class
            Max **wrong** snippets per **gold** class (errors from different true categories).
        fold_roles
            If set (e.g. ``{'middle'}``), only run XAI for those keys in ``selected_folds``.
        shap_algorithm, shap_max_evals
            Used when ``method='shap'`` (``partition`` ≈ Shapley-style grouping; slower → more stable).
        """
        assert method in ('attention', 'saliency', 'shap')
        if method == 'attention':
            cmap, method_fn = 'Blues', self.get_attention
            method_label = 'Attention (last layer, CLS, avg heads)'
        elif method == 'saliency':
            cmap, method_fn = 'Oranges', self.get_saliency
            method_label = 'Gradient Saliency'
        else:
            cmap, method_fn = 'Purples', None
            method_label = (
                f"SHAP ({shap_algorithm}, |φ| for predicted class; "
                f"Text masker, max_evals={shap_max_evals})"
            )

        fold_items = selected_folds.items()
        if fold_roles is not None:
            fold_items = [(r, fi) for r, fi in fold_items if r in fold_roles]

        for role, fold_info in fold_items:
            fold_idx = int(fold_info['fold_idx'])
            did = dialogue_for_fold(
                self.experiment_name,
                fold_idx,
                results_path=self.results_path,
                strict=True,
            )
            print(f"Loading {self.experiment_name} — {role.upper()} fold "
                  f"({did})  [{method_label}]...")
            model   = self.load_model(fold_idx)
            all_df  = self.get_snippets(fold_info, loader)

            # First pass: run live model on all snippets to get real predictions
            live_preds = []
            for text in all_df['model_text']:
                with torch.no_grad():
                    text_emb = self._quick_embed(model, text)
                live_preds.append(int(model.head(text_emb).argmax().item()))

            all_df = all_df.copy()
            all_df['live_pred'] = live_preds
            all_df['correct']   = all_df['live_pred'] == all_df['label']

            correct_df = self._stratified_by_true_label(
                all_df[all_df['correct']],
                max_per_class=correct_per_true_class,
            )
            wrong_df = self._stratified_by_true_label(
                all_df[~all_df['correct']],
                max_per_class=wrong_per_true_class,
            )

            print(
                f"  [{role}] XAI samples: {len(correct_df)} correct "
                f"({correct_df['label'].nunique()} gold classes), "
                f"{len(wrong_df)} wrong ({wrong_df['label'].nunique()} gold classes) "
                f"(cap {correct_per_true_class}/class correct, {wrong_per_true_class}/class wrong)."
            )

            samples = pd.concat([correct_df, wrong_df], ignore_index=True)
            n       = len(samples)

            shap_explainer = None
            if method == 'shap':
                shap_explainer = self._create_shap_text_explainer(
                    model, algorithm=shap_algorithm
                )

            fig = plt.figure(figsize=(22, n * 3.5))
            outer = gridspec.GridSpec(n, 1, figure=fig, hspace=0.55)

            for sample_i, (_, row) in enumerate(samples.iterrows()):
                if method == 'shap':
                    assert shap_explainer is not None
                    words, scores, pred_cls = self._explain_shap_snippet(
                        shap_explainer,
                        model,
                        row['model_text'],
                        max_evals=shap_max_evals,
                    )
                else:
                    assert method_fn is not None
                    result = method_fn(model, row['model_text'])
                    words  = result[0]
                    scores = result[1]
                    pred_cls = (
                        result[2] if method == 'saliency' else int(row['live_pred'])
                    )
                true_name = label_names[int(row['label'])]
                pred_name = label_names[pred_cls]
                correct   = bool(row['correct'])
                status    = '✓ Correct' if correct else '✗ Wrong'

                # --- compute how many rows the heatmap needs ---
                n_text_rows = max(1, int(np.ceil(len(words) / words_per_row)))
                row_height  = max(1.8, n_text_rows * 0.55)

                inner = gridspec.GridSpecFromSubplotSpec(
                    1, 2, subplot_spec=outer[sample_i],
                    width_ratios=[2.5, 1], wspace=0.08
                )
                ax_heat = fig.add_subplot(inner[0])
                ax_bar  = fig.add_subplot(inner[1])

                self._plot_word_heatmap(words, scores, ax_heat,
                                        cmap_name=cmap, words_per_row=words_per_row)
                self._plot_top_words_bar(words, scores, ax_bar,
                                         top_n=min(top_n, len(words)), cmap_name=cmap)

                title = (f"[{status}]  True: {true_name}  |  Pred: {pred_name}  "
                         f"({len(words)} words)")
                ax_heat.set_title(title, fontsize=10, pad=6,
                                  color='#1a6b2e' if correct else '#8b0000',
                                  fontweight='bold')

            fig.suptitle(
                f"{role.upper()} fold — {did} "
                f"(F1={fold_info['score']})  |  {method_label}  "
                f"[{self.experiment_name}]",
                fontsize=12, fontweight='bold', y=1.01
            )
            plt.tight_layout()
            fname = FIGURES_DIR / f'{self.experiment_name}_{method}_{role}.png'
            plt.savefig(str(fname), dpi=150, bbox_inches='tight')
            plt.show()
            del model

    def _quick_embed(self, model, snippet):
        """Run forward pass and return mean-pooled text embedding."""
        enc = self.tokenizer(snippet, return_tensors='pt',
                             truncation=True, max_length=128, padding=False)
        enc = self._enc_to_model_device(enc, model)
        with torch.no_grad():
            out      = model.model(input_ids=enc['input_ids'],
                                   attention_mask=enc['attention_mask'])
            mask     = enc['attention_mask'].float()
            text_emb = (out.last_hidden_state * mask[:, :, None]).sum(dim=1)
            text_emb = model.dropout(text_emb) / mask.sum(dim=1)[:, None]
        return text_emb
