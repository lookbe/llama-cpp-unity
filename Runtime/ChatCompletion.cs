using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using UnityEngine;

namespace LlamaCpp
{
    public class ChatCompletion : BackgroundRunner
    {
        [Header("Model")]
        [Tooltip("GGUF model absolute path")]
        public string modelPath = string.Empty;

        [Header("Context")]
        [TextArea(10, 20)]
        public string systemPrompt = string.Empty;
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
        IntPtr _chatTemplate = IntPtr.Zero;

        Sampling.common_sampler common_sampler;

        uint n_ctx = 4096;
        int n_batch = 512;
        int n_past = 0;
        int n_keep = 1;

        protected List<Chat.common_chat_msg> chat_msgs = new();

        private void PostResponseStream(string response)
        {
            unityContext?.Post(_ => OnResponseStreamed?.Invoke(response), null);
        }

        private void PostResponse(string response)
        {
            unityContext?.Post(_ => OnResponseGenerated?.Invoke(response), null);
        }

        public void InitModel()
        {
            if (string.IsNullOrEmpty(modelPath))
            {
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

                _chatTemplate = Native.llama_model_chat_template(_llamaModel, null);
                string template = Marshal.PtrToStringUTF8(_chatTemplate);

                Sampling.common_params_sampling sampling = Sampling.common_params_sampling.create_default();
                sampling.temp = temperature;
                sampling.top_k = topK;
                sampling.top_p = topP;
                sampling.min_p = minP;
                sampling.penalty_repeat = repeatPenalty;

                common_sampler = Sampling.common_sampler_init(_llamaModel, sampling);

                string intialPrompt = "";
                if (!string.IsNullOrEmpty(systemPrompt))
                {
                    var msg = new Chat.common_chat_msg() { role = "system", content = systemPrompt };
                    intialPrompt = Chat.common_chat_format_single(_chatTemplate, chat_msgs.ToArray(), msg , false);
                    chat_msgs.Add(msg);
                }

                // this step quite crucial to keep consistent generation
                int[] initial_token = Common.common_tokenize(_llamaVocab, intialPrompt, true, true);
                for (int i = 0; i < initial_token.Length; i++)
                {
                    Sampling.common_sampler_accept(ref common_sampler, initial_token[i], false);
                }

                embd.AddRange(initial_token);

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
            Sampling.common_sampler_free(ref common_sampler);

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

        private class PromptPayload : IBackgroundPayload
        {
            public string Prompt;
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
            RunBackground(new PromptPayload() { Prompt = prompt }, RunPrompt);
        }

        List<int[]> batch_split(int[] source)
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

        void try_decode(int[] token_list)
        {
            // split and decode
            List<int[]> batches = batch_split(token_list);
            for (int i = 0; i < batches.Count; i++)
            {
                int[] current_batches = batches[i];
                if (n_past + current_batches.Length > n_ctx)
                {
                    Debug.Log("context shift");

                    int n_left = n_past - n_keep;
                    int n_discard = n_left / 2;

                    Native.llama_memory_seq_rm(Native.llama_get_memory(_llamaContext), 0, n_keep, n_keep + n_discard);
                    Native.llama_memory_seq_add(Native.llama_get_memory(_llamaContext), 0, n_keep + n_discard, n_past, -n_discard);

                    n_past -= n_discard;
                }

                n_past += current_batches.Length;
                Native.llama_batch llama_batch = Native.llama_batch_get_one(current_batches, current_batches.Length);
                int decode_result = Native.llama_decode(_llamaContext, llama_batch);
                if (decode_result != 0)
                {
                    throw new Exception($"decode error result {decode_result} used ctx {n_past}");
                }
            }
        }

        List<int> embd = new();
        List<int> embd_input = new();

        unsafe void RunPrompt(PromptPayload payload, CancellationToken cts)
        {
            string prompt = payload.Prompt;
            string response = "";
            int[] token_list = new int[0];

            try
            {
                while (true)
                {
                    if (!string.IsNullOrEmpty(prompt))
                    {
                        var input_msg = new Chat.common_chat_msg() { role = "user", content = prompt };

                        string fmt_prompt = Chat.common_chat_format_single(_chatTemplate, chat_msgs.ToArray(), input_msg, true);
                        chat_msgs.Add(input_msg);

                        token_list = Common.common_tokenize(_llamaVocab, fmt_prompt, false, true);
                        embd_input.AddRange(token_list);

                        Sampling.common_sampler_reset(ref common_sampler);
                        prompt = string.Empty;
                    }

                    if (embd.Count > 0)
                    {
                        try_decode(embd.ToArray());
                    }

                    embd.Clear();

                    if (embd_input.Count == 0)
                    {
                        int last_token = Sampling.common_sampler_sample(ref common_sampler, _llamaContext, -1, false);
                        string piece = Common.common_token_to_piece(_llamaVocab, last_token, false);

                        response += piece;
                        PostResponseStream(piece);

                        Sampling.common_sampler_accept(ref common_sampler, last_token, true);
                        embd.Add(last_token);
                    }
                    else
                    {
                        int[] input_token = embd_input.ToArray();
                        for (int i = 0; i < token_list.Length; i++)
                        {
                            Sampling.common_sampler_accept(ref common_sampler, input_token[i], false);
                        }
                        embd.AddRange(input_token);
                        embd_input.Clear();
                    }

                    // break here since we want to decode eog token before stop generate
                    if (Native.llama_vocab_is_eog(_llamaVocab, Sampling.common_sampler_last(ref common_sampler)))
                    {
                        chat_msgs.Add(new Chat.common_chat_msg() { role = "assistant", content = response });
                        break;
                    }

                    if (cts.IsCancellationRequested)
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
                PostResponse(response);
                PostStatus(ModelStatus.Ready);
            }
        }
    }
}
