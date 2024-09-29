import importlib
import os
from typing import Optional

from peft import AutoPeftModelForCausalLM, PeftConfig, MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from peft.utils.constants import TOKENIZER_CONFIG_NAME
from peft.utils.other import check_file_exists_on_hf_hub
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration


class AutoPeftModelForCausalLMWithResizedWTE(AutoPeftModelForCausalLM):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        **kwargs,
    ):
        r"""
        A wrapper around all the preprocessing steps a user needs to perform in order to load a PEFT model. The kwargs
        are passed along to `PeftConfig` that automatically takes care of filtering the kwargs of the Hub methods and
        the config object init.
        """
        peft_config = PeftConfig.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        base_model_path = peft_config.base_model_name_or_path

        task_type = getattr(peft_config, "task_type", None)

        if (
            task_type == "QWEN_CONDITIONAL_GENERATION"
            or "Qwen2-VL" in peft_config.base_model_name_or_path
        ):
            target_class = Qwen2VLForConditionalGeneration
        elif cls._target_class is not None:
            target_class = cls._target_class
        elif cls._target_class is None and task_type is not None:
            # this is only in the case where we use `AutoPeftModel`
            raise ValueError(
                "Cannot use `AutoPeftModel` with a task type, please use a specific class for your task type. (e.g. `AutoPeftModelForCausalLM` for `task_type='CAUSAL_LM'`)"
            )

        if task_type is not None:
            if (
                task_type == "QWEN_CONDITIONAL_GENERATION"
                or "Qwen2-VL" in peft_config.base_model_name_or_path
            ):
                expected_target_class = Qwen2VLForConditionalGeneration
            else:
                expected_target_class = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[task_type]
                if cls._target_peft_class.__name__ != expected_target_class.__name__:
                    raise ValueError(
                        f"Expected target PEFT class: {expected_target_class.__name__}, but you have asked for: {cls._target_peft_class.__name__}"
                        " make sure that you are loading the correct model for your task type."
                    )
        elif (
            task_type is None and getattr(peft_config, "auto_mapping", None) is not None
        ):
            auto_mapping = getattr(peft_config, "auto_mapping", None)
            base_model_class = auto_mapping["base_model_class"]
            parent_library_name = auto_mapping["parent_library"]

            parent_library = importlib.import_module(parent_library_name)
            target_class = getattr(parent_library, base_model_class)
        else:
            raise ValueError(
                "Cannot infer the auto class from the config, please make sure that you are loading the correct model for your task type."
            )

        base_model = target_class.from_pretrained(base_model_path, **kwargs)

        tokenizer_exists = False
        if os.path.exists(
            os.path.join(pretrained_model_name_or_path, TOKENIZER_CONFIG_NAME)
        ):
            tokenizer_exists = True
        else:
            token = kwargs.get("token", None)
            if token is None:
                token = kwargs.get("use_auth_token", None)

            # tokenizer_exists = check_file_exists_on_hf_hub(
            #     repo_id=pretrained_model_name_or_path,
            #     filename=TOKENIZER_CONFIG_NAME,
            #     revision=kwargs.get("revision", None),
            #     repo_type=kwargs.get("repo_type", None),
            #     token=token,
            # )

            tokenizer_exists = check_file_exists_on_hf_hub(
                repo_id=base_model_path,
                filename=TOKENIZER_CONFIG_NAME,
                revision=kwargs.get("revision", None),
                repo_type=kwargs.get("repo_type", None),
                token=token,
            )

        if tokenizer_exists:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=kwargs.get("trust_remote_code", False),
            )
            base_model.resize_token_embeddings(len(tokenizer))

        if len(tokenizer) != base_model.get_input_embeddings().weight.size(0):
            base_model.resize_token_embeddings(len(tokenizer))

        return cls._target_peft_class.from_pretrained(
            base_model,
            pretrained_model_name_or_path,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
            config=config,
            **kwargs,
        )
