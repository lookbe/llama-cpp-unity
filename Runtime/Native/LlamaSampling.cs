using System;

namespace LlamaCpp
{
    public unsafe static class Sampling
    {
        public struct common_params_sampling
        {
            public uint seed;
            public int n_prev;
            public int n_probs;
            public int min_keep;
            public int top_k;
            public float top_p;
            public float min_p;
            public float xtc_probability;
            public float xtc_threshold;
            public float typ_p;
            public float temp;
            public float dynatemp_range;
            public float dynatemp_exponent;
            public int penalty_last_n;
            public float penalty_repeat;
            public float penalty_freq;
            public float penalty_present;
            public float top_n_sigma;

            public static common_params_sampling create_default()
            {
                return new common_params_sampling()
                {
                    seed = 0xFFFFFFFF,
                    n_prev = 64,
                    n_probs = 0,
                    min_keep = 0,
                    top_k = 40,
                    top_p = 0.95f,
                    min_p = 0.05f,
                    xtc_probability = 0.00f,
                    xtc_threshold = 0.10f,
                    typ_p = 1.00f,
                    temp = 0.80f,
                    dynatemp_range = 0.00f,
                    dynatemp_exponent = 1.00f,
                    penalty_last_n = 64,
                    penalty_repeat = 1.00f,
                    penalty_freq = 0.00f,
                    penalty_present = 0.00f,
                    top_n_sigma = -1.00f,
                };
            }
        }

        public struct common_sampler
        {
            public IntPtr llama_grammar;
            public IntPtr llama_chain;

            public Native.llama_token_data[] token_data;
            public Native.llama_token_data_array token_data_array;
            public int last_token;
            public unsafe void set_logits(IntPtr ctx, int idx)
            {
                float* logits = (float*)Native.llama_get_logits_ith(ctx, idx);

                for (int i = 0; i < token_data.Length; i++)
                {
                    token_data[i].id = i;
                    token_data[i].logit = logits[i];
                    token_data[i].p = 0.0f;
                }
            }
        }

        public static common_sampler common_sampler_init(IntPtr model, common_params_sampling param_sampling)
        {
            common_sampler result = new common_sampler();

            IntPtr llama_vocab = Native.llama_model_get_vocab(model);
            int n_vocab = Native.llama_vocab_n_tokens(llama_vocab);

            Native.llama_sampler_chain_params sparams = Native.llama_sampler_chain_default_params();

            result.llama_grammar = Native.llama_sampler_init_grammar(llama_vocab, "", "root");
            result.llama_chain = Native.llama_sampler_chain_init(sparams);
            result.token_data = new Native.llama_token_data[n_vocab];
            result.token_data_array = new Native.llama_token_data_array();

            Native.llama_sampler_chain_add(result.llama_chain, Native.llama_sampler_init_top_k(param_sampling.top_k));
            Native.llama_sampler_chain_add(result.llama_chain, Native.llama_sampler_init_top_p(param_sampling.top_p, param_sampling.min_keep));
            Native.llama_sampler_chain_add(result.llama_chain, Native.llama_sampler_init_min_p(param_sampling.min_p, param_sampling.min_keep));
            Native.llama_sampler_chain_add(result.llama_chain, Native.llama_sampler_init_typical(param_sampling.typ_p, param_sampling.min_keep));
            Native.llama_sampler_chain_add(result.llama_chain, Native.llama_sampler_init_temp_ext(param_sampling.temp, param_sampling.dynatemp_range, param_sampling.dynatemp_exponent));
            Native.llama_sampler_chain_add(result.llama_chain, Native.llama_sampler_init_xtc(param_sampling.xtc_probability, param_sampling.xtc_threshold, param_sampling.min_keep, param_sampling.seed));
            Native.llama_sampler_chain_add(result.llama_chain, Native.llama_sampler_init_penalties(param_sampling.penalty_last_n, param_sampling.penalty_repeat, param_sampling.penalty_freq, param_sampling.penalty_present));
            Native.llama_sampler_chain_add(result.llama_chain, Native.llama_sampler_init_top_n_sigma(param_sampling.top_n_sigma));
            Native.llama_sampler_chain_add(result.llama_chain, Native.llama_sampler_init_dist(param_sampling.seed)); // this must be last

            return result;
        }

        public static void common_sampler_free(ref common_sampler sampler)
        {
            if (sampler.llama_grammar != IntPtr.Zero)
            {
                Native.llama_sampler_free(sampler.llama_grammar);
                sampler.llama_grammar = IntPtr.Zero;
            }

            if (sampler.llama_chain != IntPtr.Zero)
            {
                Native.llama_sampler_free(sampler.llama_chain);
                sampler.llama_chain = IntPtr.Zero;
            }
        }

        public static void common_sampler_accept(ref common_sampler sampler, int token, bool accept_grammar)
        {
            if (accept_grammar)
            {
                Native.llama_sampler_accept(sampler.llama_grammar, token);
            }

            Native.llama_sampler_accept(sampler.llama_chain, token);
            sampler.last_token = token;
        }

        public static void common_sampler_reset(ref common_sampler sampler)
        {
            Native.llama_sampler_reset(sampler.llama_grammar);
            Native.llama_sampler_reset(sampler.llama_chain);
        }

        public static int common_sampler_last(ref common_sampler sampler)
        {
            return sampler.last_token;
        }

        public unsafe static int common_sampler_sample(ref common_sampler sampler, IntPtr ctx, int idx, bool grammar_first)
        {
            using Util.memory_pin_manager memory_manager = new Util.memory_pin_manager();
            {
                sampler.set_logits(ctx, idx);
                sampler.token_data_array.data = memory_manager.Pin(sampler.token_data.AsMemory());
                sampler.token_data_array.size = sampler.token_data.Length;
                sampler.token_data_array.selected = -1;
                sampler.token_data_array.sorted = 0;

                if (grammar_first)
                {
                    Native.llama_sampler_apply(sampler.llama_grammar, ref sampler.token_data_array);
                }

                Native.llama_sampler_apply(sampler.llama_chain, ref sampler.token_data_array);

                int id = sampler.token_data[sampler.token_data_array.selected].id;
                if (grammar_first)
                {
                    return id;
                }

                Native.llama_token_data[] single_token_data = new Native.llama_token_data[] { new Native.llama_token_data() { id = id, logit = 1.0f, p = 0.0f } };
                Native.llama_token_data_array single_token_data_array = new Native.llama_token_data_array();
                single_token_data_array.data = memory_manager.Pin(single_token_data.AsMemory());
                single_token_data_array.size = 1;
                single_token_data_array.selected = -1;
                single_token_data_array.sorted = 0;

                Native.llama_sampler_apply(sampler.llama_grammar, ref single_token_data_array);

                bool is_valid = single_token_data[0].logit != float.NegativeInfinity;
                if (is_valid)
                {
                    return id;
                }

                sampler.set_logits(ctx, idx);
                sampler.token_data_array.data = memory_manager.Pin(sampler.token_data.AsMemory());
                sampler.token_data_array.size = sampler.token_data.Length;
                sampler.token_data_array.selected = -1;
                sampler.token_data_array.sorted = 0;

                Native.llama_sampler_apply(sampler.llama_grammar, ref sampler.token_data_array);
                Native.llama_sampler_apply(sampler.llama_chain, ref sampler.token_data_array);

                return sampler.token_data[sampler.token_data_array.selected].id;
            }
        }
    }
}
