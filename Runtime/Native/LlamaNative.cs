using System;
using System.Runtime.InteropServices;

namespace LlamaCpp
{
    public unsafe static class Native
    {
        private const string LlamaDll = "llama";

        [StructLayout(LayoutKind.Sequential)]
        public unsafe struct llama_batch
        {
            public int n_tokens;
            public int* token; // const int32_t*
            public float* embd; // float*
            public int* pos; // const int32_t*
            public int* n_seq_id; // const int32_t*
            public int** seq_id; // const int32_t**
            public byte* logits; // int8_t*
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public delegate bool llama_progress_callback(float progress, IntPtr user_data);

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_model_params
        {
            // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
            public IntPtr devices;

            // NULL-terminated list of buffer types to use for tensors that match a pattern
            public IntPtr tensor_buft_overrides;

            public int n_gpu_layers; // number of layers to store in VRAM
            public int split_mode; // how to split the model across multiple GPUs

            // the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
            public int main_gpu;

            // proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
            public IntPtr tensor_split;

            // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
            // If the provided progress_callback returns true, model loading continues.
            // If it returns false, model loading is immediately aborted.
            public llama_progress_callback progress_callback;

            // context pointer passed to the progress callback
            public IntPtr progress_callback_user_data;

            // override key-value pairs of the model meta data
            public IntPtr kv_overrides;

            // Keep the booleans together to avoid misalignment during copy-by-value.
            [MarshalAs(UnmanagedType.I1)] public bool vocab_only;      // only load the vocabulary, no weights
            [MarshalAs(UnmanagedType.I1)] public bool use_mmap;        // use mmap if possible
            [MarshalAs(UnmanagedType.I1)] public bool use_mlock;       // force system to keep model in RAM
            [MarshalAs(UnmanagedType.I1)] public bool check_tensors;   // validate model tensor data
            [MarshalAs(UnmanagedType.I1)] public bool use_extra_bufts; // use extra buffer types (used for weight repacking)
            [MarshalAs(UnmanagedType.I1)] public bool no_host;         // bypass host buffer allowing extra buffers to be used
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public delegate bool ggml_backend_sched_eval_callback(IntPtr ggml_tensor, [MarshalAs(UnmanagedType.I1)] bool ask, IntPtr user_data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public delegate bool ggml_abort_callback(IntPtr data);

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_context_params
        {
            public uint n_ctx;             // text context, 0 = from model
            public uint n_batch;           // logical maximum batch size that can be submitted to llama_decode
            public uint n_ubatch;          // physical maximum batch size
            public uint n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
            public int n_threads;         // number of threads to use for generation
            public int n_threads_batch;   // number of threads to use for batch processing

            public int rope_scaling_type; // RoPE scaling type, from `enum llama_rope_scaling_type`
            public int pooling_type;      // whether to pool (sum) embedding results by sequence id
            public int attention_type;    // attention type to use for embeddings
            public int flash_attn_type;   // when to enable Flash Attention

            // ref: https://github.com/ggml-org/llama.cpp/pull/2054
            public float rope_freq_base;   // RoPE base frequency, 0 = from model
            public float rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
            public float yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
            public float yarn_attn_factor; // YaRN magnitude scaling factor
            public float yarn_beta_fast;   // YaRN low correction dim
            public float yarn_beta_slow;   // YaRN high correction dim
            public uint yarn_orig_ctx;    // YaRN original context size
            public float defrag_thold;     // [DEPRECATED] defragment the KV cache if holes/size > thold, <= 0 disabled (default)

            public ggml_backend_sched_eval_callback cb_eval;
            public IntPtr cb_eval_user_data;

            public int type_k; // data type for K cache [EXPERIMENTAL]
            public int type_v; // data type for V cache [EXPERIMENTAL]

            // Abort callback
            // if it returns true, execution of llama_decode() will be aborted
            // currently works only with CPU execution
            public ggml_abort_callback abort_callback;
            public IntPtr abort_callback_data;

            // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
            [MarshalAs(UnmanagedType.I1)] public bool embeddings;  // if true, extract embeddings (together with logits)
            [MarshalAs(UnmanagedType.I1)] public bool offload_kqv; // offload the KQV ops (including the KV cache) to GPU
            [MarshalAs(UnmanagedType.I1)] public bool no_perf;     // measure performance timings
            [MarshalAs(UnmanagedType.I1)] public bool op_offload;  // offload host tensor operations to device
            [MarshalAs(UnmanagedType.I1)] public bool swa_full;    // use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
                                                                   // NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
                                                                   //       ref: https://github.com/ggml-org/llama.cpp/pull/13845#issuecomment-2924800573
            [MarshalAs(UnmanagedType.I1)] public bool kv_unified;  // use a unified buffer across the input sequences when computing the attention
                                                                   // try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix
                                                                   // ref: https://github.com/ggml-org/llama.cpp/pull/14363
        };

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_sampler_chain_params
        {
            [MarshalAs(UnmanagedType.I1)] public bool no_perf; // whether to measure performance timings
        }

        [StructLayout(LayoutKind.Sequential)]
        public unsafe struct llama_chat_message
        {
            public IntPtr role;
            public IntPtr content;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_token_data
        {
            public int id;      // llama_token = int
            public float logit;
            public float p;
        }

        [StructLayout(LayoutKind.Sequential)]
        public unsafe struct llama_token_data_array
        {
            public IntPtr data;     // llama_token_data*
            public long size;      // size_t
            public long selected;   // int64_t
            public byte sorted;
        }

        // backend
        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_backend_init();

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_backend_free();

        // model
        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_model_params llama_model_default_params();

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_model_load_from_file([MarshalAs(UnmanagedType.LPUTF8Str)] string path_model, llama_model_params model_params);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_model_free(IntPtr model);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_model_has_encoder(IntPtr model);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_model_has_decoder(IntPtr model);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_model_decoder_start_token(IntPtr model);

        // context
        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_context_params llama_context_default_params();

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_init_from_model(IntPtr model, llama_context_params params_ctx);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_free(IntPtr ctx);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint llama_n_ctx(IntPtr ctx);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_get_model(IntPtr ctx);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_set_warmup(IntPtr ctx, [MarshalAs(UnmanagedType.I1)] bool warmup);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_synchronize(IntPtr ctx);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_perf_context_reset(IntPtr ctx);


        // memory
        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_get_memory(IntPtr ctx);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_memory_seq_pos_max(IntPtr mem, int seq_id);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_memory_seq_rm(IntPtr mem, int seq_id, int p0, int p1);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_memory_seq_add(IntPtr mem, int seq_id, int p0, int p1, int delta);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_memory_clear(IntPtr mem, [MarshalAs(UnmanagedType.I1)] bool data);

        // token
        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_tokenize(IntPtr vocab, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, int text_len, [MarshalAs(UnmanagedType.LPArray)] int[] tokens, int n_tokens_max, [MarshalAs(UnmanagedType.I1)] bool add_special, [MarshalAs(UnmanagedType.I1)] bool parse_special);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_token_to_piece(IntPtr vocab, int token, [MarshalAs(UnmanagedType.LPArray)] byte[] buf, int length, int lstrip, [MarshalAs(UnmanagedType.I1)] bool special);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_batch llama_batch_get_one([MarshalAs(UnmanagedType.LPArray)] int[] tokens, int n_tokens);

        // encode decode
        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_encode(IntPtr ctx, llama_batch batch);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_decode(IntPtr ctx, llama_batch batch);

        //Grammar
        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_grammar(IntPtr vocab, [MarshalAs(UnmanagedType.LPUTF8Str)] string grammar_str, [MarshalAs(UnmanagedType.LPUTF8Str)] string grammar_root);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_get_logits_ith(IntPtr ctx, int idx);

        // sampler
        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_sampler_chain_params llama_sampler_chain_default_params();

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_chain_init(llama_sampler_chain_params sampler_params);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sampler_free(IntPtr sampler);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sampler_chain_add(IntPtr chain, IntPtr smpl);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_top_k(int k);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_top_p(float p, long min_keeo);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_min_p(float p, int min_keep);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_typical(float p, int min_keep);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_temp_ext(float t, float delta, float exponent);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_temp(float t);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_xtc(float p, float t, int min_keep, uint seed);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_top_n_sigma(float n);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_dist(uint seed);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_penalties(int penalty_last_n, float penalty_repeat, float penalty_freq, float penalty_present);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_infill(IntPtr vocab);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_sampler_sample(IntPtr smpl, IntPtr ctx, int idx);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sampler_reset(IntPtr smpl);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sampler_accept(IntPtr smpl, int token);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sampler_apply(IntPtr smpl, ref llama_token_data_array cur_p);

        // vocab
        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_model_get_vocab(IntPtr model);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_vocab_n_tokens(IntPtr vocab);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_vocab_is_eog(IntPtr vocab, int token);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_vocab_is_control(IntPtr vocab, int token);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_vocab_bos(IntPtr vocab); // beginning-of-sentence

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_vocab_eos(IntPtr vocab); // end-of-sentence

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_vocab_eot(IntPtr vocab); // end-of-turn

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_vocab_sep(IntPtr vocab); // sentence separator

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_vocab_nl(IntPtr vocab); // next-line

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_vocab_pad(IntPtr vocab); // padding

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_vocab_mask(IntPtr vocab); // mask

        // chat
        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_model_chat_template(IntPtr model, [MarshalAs(UnmanagedType.LPUTF8Str)] string name);

        [DllImport(LlamaDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_chat_apply_template(IntPtr tmpl, [MarshalAs(UnmanagedType.LPArray)] llama_chat_message[] chat, int n_msg, [MarshalAs(UnmanagedType.I1)] bool add_ass, [MarshalAs(UnmanagedType.LPArray)] byte[] buf, int length);
    }

}
