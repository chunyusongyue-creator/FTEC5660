# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.



## Homework 1 solution:

### Chain Design

```mermaid
flowchart TD
    A["🧾 Receipt Image"]

    A --> B["① Extractor<br/>final_payment<br/>subtotal<br/>discounts"]
    A --> C["② Blind Independent Reviewer<br/>final_payment<br/>subtotal<br/>discounts<br/>without_discount"]

    B --> D{"Do both independent<br/>results agree?"}
    C --> D

    D -- "Yes" --> E["Accept validated result"]

    D -- "No / one pass fails" --> F["③ Judge<br/>re-check original receipt<br/>final_payment<br/>subtotal<br/>discounts<br/>without_discount"]

    A --> F

    E --> G["Final receipt result"]
    F --> G

    G --> H["Python Decimal Aggregation"]

    H --> I["Query 1<br/>Sum validated final_payment"]
    H --> J["Query 2<br/>Sum subtotal + discounts<br/>Exclude rounding"]

    I --> K["💰 Final HKD Answers"]
    J --> K

    classDef input fill:#eef4ff,stroke:#4a6fa5,stroke-width:2px,color:#111;
    classDef agent fill:#f3efff,stroke:#7251b5,stroke-width:2px,color:#111;
    classDef decision fill:#fff4d6,stroke:#c28b00,stroke-width:2px,color:#111;
    classDef process fill:#eaf7ee,stroke:#3f8f5f,stroke-width:2px,color:#111;
    classDef output fill:#fff0f3,stroke:#b85c72,stroke-width:2px,color:#111;

    class A input;
    class B,C,F agent;
    class D decision;
    class E,G,H,I,J process;
    class K output;
```

My solution uses a multi-stage LangChain pipeline with the required `deepseek-v4-flash-vision-exp` vision model. For each receipt, an Extractor first identifies the `final_payment`, printed `subtotal`, and all qualifying discount amounts, while a Blind Independent Reviewer separately examines the original receipt image from scratch and independently extracts the same information together with `without_discount`. The two outputs are then compared for both queries. If they agree, the validated result is accepted directly; if they disagree or one pass cannot be validated, a Judge re-examines the original receipt and determines the final `final_payment`, `subtotal`, discounts, and `without_discount` values. The final receipt-level results are aggregated using Python `Decimal` arithmetic: Query 1 sums the validated `final_payment` values after rounding, while Query 2 sums `subtotal + all qualifying discounts` and excludes rounding adjustments. This design provides independent verification and conditional recovery while keeping the final response for each query as a single HKD amount.