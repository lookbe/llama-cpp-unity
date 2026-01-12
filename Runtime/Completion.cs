using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEngine;

namespace LlamaCpp
{
    public class Completion : BackgroundRunner
    {
        [Header("Model")]
        [Tooltip("GGUF model absolute path")]
        public string modelPath = string.Empty;

        [Header("Context")]
        public uint contextLength = 4096;

        [Header("Sampling")]
        public float temperature = 0.8f;
        public int topK = 40;
        public float topP = 0.95f;
        public float minP = 0.05f;
        public float repeatPenalty = 1.0f;

        // Define a delegate (or use Action<T>)
        public delegate void StatusChangedDelegate(ModelStatus status);
        public event StatusChangedDelegate OnStatusChanged;

        private ModelStatus _status = ModelStatus.Init;

        // Public getter, no public setter
        public ModelStatus status
        {
            get => _status;
            protected set
            {
                if (_status != value)
                {
                    _status = value;
                    OnStatusChanged?.Invoke(_status);
                }
            }
        }

        protected void PostStatus(ModelStatus newStatus)
        {
            unityContext?.Post(_ => status = newStatus, null);
        }

        private void Start()
        {
            Backend.Init();
        }

        async void OnDestroy()
        {
            await BackgroundStop();
            FreeModel();
            Backend.Free();
        }

        // Define a delegate (or use Action<T>)
        public delegate void ResponseStreamDelegate(string response);
        public event ResponseStreamDelegate OnResponseStreamed;

        public delegate void ResponseGeneratedDelegate(string response);
        public event ResponseGeneratedDelegate OnResponseGenerated;

        protected IntPtr _llamaModel = IntPtr.Zero;
        protected IntPtr _llamaVocab = IntPtr.Zero;
        protected IntPtr _llamaContext = IntPtr.Zero;

        uint n_ctx = 4096;
        int n_batch = 512;

        protected void PostResponseStream(string response)
        {
            unityContext?.Post(_ => OnResponseStreamed?.Invoke(response), null);
        }

        protected void PostResponse(string response)
        {
            unityContext?.Post(_ => OnResponseGenerated?.Invoke(response), null);
        }

        public void InitModel()
        {
            if (string.IsNullOrEmpty(modelPath))
            {
                Debug.LogError("model path not set");
                return;
            }

            if (_status != ModelStatus.Init)
            {
                Debug.LogError("invalid status");
                return;
            }

            status = ModelStatus.Loading;
            RunBackground(RunInitModel);
        }

        void RunInitModel(CancellationToken cts)
        {
            try
            {
                Debug.Log($"Load model at {modelPath}");

                Native.llama_model_params model_params = Native.llama_model_default_params();
                _llamaModel = Native.llama_model_load_from_file(modelPath, model_params);
                if (_llamaModel == IntPtr.Zero)
                {
                    throw new System.Exception("unable to load model");
                }

                Native.llama_context_params ctx_params = Native.llama_context_default_params();
                ctx_params.n_ctx = contextLength;
                _llamaContext = Native.llama_init_from_model(_llamaModel, ctx_params);
                if (_llamaContext == IntPtr.Zero)
                {
                    throw new("failed to create the llama_context");
                }

                n_ctx = Native.llama_n_ctx(_llamaContext);

                _llamaVocab = Native.llama_model_get_vocab(_llamaModel);

                Debug.Log("Load model done");

                PostStatus(ModelStatus.Ready);
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"An unexpected error occurred: {ex.Message}");

                FreeModel();
                PostStatus(ModelStatus.Init);
            }
        }

        void FreeModel()
        {
            if (_llamaContext != IntPtr.Zero)
            {
                Native.llama_free(_llamaContext);
                _llamaContext = IntPtr.Zero;
            }

            if (_llamaModel != IntPtr.Zero)
            {
                Native.llama_model_free(_llamaModel);
                _llamaModel = IntPtr.Zero;
            }
        }

        protected class CompletionPayload : IBackgroundPayload
        {
            public float Temp = 0.8f;
            public int TopK = 40;
            public float TopP = 0.95f;
            public float MinP = 0.05f;
            public float RepeatPenalty = 1.0f;
        }

        protected class PromptPayload : CompletionPayload
        {
            public string Prompt = string.Empty;
        }

        public void Prompt(string prompt)
        {
            if (string.IsNullOrEmpty(prompt))
            {
                return;
            }

            if (_llamaContext == IntPtr.Zero)
            {
                Debug.LogError("invalid context");
                return;
            }

            if (_llamaModel == IntPtr.Zero)
            {
                Debug.LogError("model not loaded");
                return;
            }

            if (status != ModelStatus.Ready)
            {
                Debug.LogError("invalid status");
                return;
            }

            status = ModelStatus.Generate;
            var payload = new PromptPayload()
            {
                Prompt = prompt,
                Temp = this.temperature,
                TopK = this.topK,
                TopP = this.topP,
                MinP = this.minP,
                RepeatPenalty = this.repeatPenalty,

            };

            RunBackground(payload, RunPrompt);
        }

        protected List<int[]> batch_split(int[] source)
        {
            var result = new List<int[]>((source.Length + n_batch - 1) / n_batch);

            for (int i = 0; i < source.Length; i += n_batch)
            {
                int length = Math.Min(n_batch, source.Length - i);
                int[] chunk = new int[length];
                Array.Copy(source, i, chunk, 0, length);
                result.Add(chunk);
            }

            return result;
        }

        protected void try_decode(int[] token_list)
        {
            // split and decode
            List<int[]> batches = batch_split(token_list);
            for (int i = 0; i < batches.Count; i++)
            {
                int[] current_batches = batches[i];
                Native.llama_batch llama_batch = Native.llama_batch_get_one(current_batches, current_batches.Length);
                int decode_result = Native.llama_decode(_llamaContext, llama_batch);
                if (decode_result != 0)
                {
                    throw new Exception($"decode error result {decode_result}");
                }
            }
        }

        public virtual int[] Tokenize(string prompt)
        {
            return Common.common_tokenize(_llamaVocab, prompt, true, true);
        }

        protected virtual bool EndGeneration(int token, int generated_token_count)
        {
            return Native.llama_vocab_is_eog(_llamaVocab, token);
        }

        public virtual void Stop()
        {
            if (cts != null)
            {
                cts.Cancel();
            }
        }

        protected class GenerationPayload : CompletionPayload
        {
            public int[] Tokens;
        }

        void RunPrompt(PromptPayload inputPayload, CancellationToken cts)
        {
            string prompt = inputPayload.Prompt;
            int[] token_list = Tokenize(prompt);

            var payload = new GenerationPayload()
            {
                Tokens = token_list.ToArray(),
                Temp = inputPayload.Temp,
                TopK = inputPayload.TopK,
                TopP = inputPayload.TopP,
                MinP = inputPayload.MinP,
                RepeatPenalty = inputPayload.RepeatPenalty,

            };

            RunGenerate(payload, cts);
        }

        protected virtual void RunGenerate(GenerationPayload payload, CancellationToken cts)
        {
            string response = "";
            int last_token = 0;
            int[] token_list = payload.Tokens;
            int generated_token_count = 0;

            Sampling.common_params_sampling sampling = Sampling.common_params_sampling.create_default();
            sampling.temp = payload.Temp;
            sampling.top_k = payload.TopK;
            sampling.top_p = payload.TopP;
            sampling.min_p = payload.MinP;
            sampling.penalty_repeat = payload.RepeatPenalty;

            Native.llama_sampler_chain_params sparams = Native.llama_sampler_chain_default_params();
            IntPtr llamaSampler = Native.llama_sampler_chain_init(sparams);

            Native.llama_sampler_chain_add(llamaSampler, Native.llama_sampler_init_top_k(sampling.top_k));
            Native.llama_sampler_chain_add(llamaSampler, Native.llama_sampler_init_top_p(sampling.top_p, sampling.min_keep));
            Native.llama_sampler_chain_add(llamaSampler, Native.llama_sampler_init_min_p(sampling.min_p, sampling.min_keep));
            Native.llama_sampler_chain_add(llamaSampler, Native.llama_sampler_init_typical(sampling.typ_p, sampling.min_keep));
            Native.llama_sampler_chain_add(llamaSampler, Native.llama_sampler_init_temp_ext(sampling.temp, sampling.dynatemp_range, sampling.dynatemp_exponent));
            Native.llama_sampler_chain_add(llamaSampler, Native.llama_sampler_init_xtc(sampling.xtc_probability, sampling.xtc_threshold, sampling.min_keep, sampling.seed));
            Native.llama_sampler_chain_add(llamaSampler, Native.llama_sampler_init_penalties(sampling.penalty_last_n, sampling.penalty_repeat, sampling.penalty_freq, sampling.penalty_present));
            Native.llama_sampler_chain_add(llamaSampler, Native.llama_sampler_init_top_n_sigma(sampling.top_n_sigma));
            Native.llama_sampler_chain_add(llamaSampler, Native.llama_sampler_init_dist(sampling.seed)); // this must be last

            try
            {
                while (true)
                {
                    if (cts.IsCancellationRequested)
                    {
                        break;
                    }

                    try_decode(token_list);

                    last_token = Native.llama_sampler_sample(llamaSampler, _llamaContext, -1);
                    string piece = Common.common_token_to_piece(_llamaVocab, last_token, false);
                    response += piece;
                    generated_token_count++;

                    PostResponseStream(piece);

                    // create next stream
                    token_list = new[] { last_token };

                    // break here since we want to decode eog token before stop generate
                    if (EndGeneration(last_token, generated_token_count))
                    {
                        break;
                    }
                }
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"An unexpected error occurred: {ex.Message}");
            }
            finally
            {
                Native.llama_memory_seq_rm(Native.llama_get_memory(_llamaContext), 0, -1, -1);

                if (llamaSampler != IntPtr.Zero)
                {
                    Native.llama_sampler_free(llamaSampler);
                    llamaSampler = IntPtr.Zero;
                }

                PostResponse(response);
                PostStatus(ModelStatus.Ready);
            }
        }
    }
}
