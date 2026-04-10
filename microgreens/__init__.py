# SPDX-License-Identifier: Apache-2.0
"""
Microgreens: Sibling MTP heads for tree-structured speculative decoding.

Modules:
    mtp_clone          - Clone the Qwen 3.5-27B MTP head into K noisy siblings
    mtp_diversity_train - Fine-tune siblings for prediction diversity
    sibling_mtp_proposer - vLLM integration for tree-structured drafting
"""
