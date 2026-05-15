* You are in a Python project that benchmarks local LLM models for accuracy.
* Create a Python script called benchmark_suite.py that will loop through and benchmark all models available in LMStudio with 'lms' executable. You can view all available models with 'lms ls'.
* You can execute each benchmark with the command 'pixi run python bench.py run --corpus {code} --model {model}', where {code} is a provided path to a Python file, and {model} is the name of the llm model in LMStudio.
* You will need to create a config file for each model in the folder 'configs/models/user' (create the folder if it does not exist) that contains the necessary parameters for the benchmark of that specific model. You will name the toml file as '{model}-{framework}-{max_tokens}.toml, where {framework} is either gguf or mlx based on the type of model.
* Use the template model toml files in 'configs/models' and find the most appropriate to use as a starting point for the LMStudio model that is being benchmarked. Create a custom copy of the model toml in 'configs/models/user' and allow the user to passed the desired max_tokens value at the top level of the script and as a CLI option. This user specified max_tokens must be updated in the toml files for all models in 'configs/models/user'.
* Run the benchmark for all models and sleep for 30 seconds between benchmarks.
* Execute 'pixi run python analysis/visualize.py' at the end to create the visual report of the benchmark results.
* Allow the user to also provide a custom path to the corpora but use 'configs/corpora' as the default path.
* Allow the user to also provide a custom path to the save all the model toml file but use 'configs/corpora' as the default path.
* Execute all Python code using the pixi environment with the command 'pixi run python'.
* Create a Markdown file with a table containing the pass, hallucinations, and bonus, and total runtime for each model that was executed in this script. Place the models as rows and pass, hallucinations, and bonus as columns.
* Test with a small model like mlx-community/qwen3.6-35b-a3b`.
* Update to allow an identical benchmack capability with ollama as we have with lmstudio. You can use 'ollama ls' to gather all the models available in ollama. Add a new CLI argument --runner to specify whether to benchmark lmstudio or ollama.
