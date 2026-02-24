import logging
from sal.config import Config
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score, score_pass_at_k
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()
    config.hub_dataset_id = "ENSEONG/score-early-MATH-500-Qwen2.5-3B-Instruct-bon"
    config.push_to_hub = True
    config.n = 64
    seeds=[0,42,64]
    for seed in seeds:
        load_subset_name= f"temp-low-0.1-high-0.8-probe-8_total-64_thresh-6_seed-{seed}"
        
        dataset = load_dataset("ENSEONG/early-MATH-500-Qwen2.5-3B-Instruct-bon",load_subset_name)
        dataset = score(dataset, config)
        dataset = score_pass_at_k(dataset, config)
        
        # push to hub
        if config.push_to_hub:
            logger.info(f"Pushing dataset to hub: {config.hub_dataset_id} (subset: {load_subset_name})")
            dataset.push_to_hub(
                repo_id=config.hub_dataset_id,
                config_name=load_subset_name,  # subset으로 지정
                private=False
            )
            logger.info(f"Successfully pushed to {config.hub_dataset_id} (config: {load_subset_name})")

    
    logger.info("Done 🔥!")

if __name__ == "__main__":
    main()
