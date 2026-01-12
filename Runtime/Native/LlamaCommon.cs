using System;
using System.Text;
using UnityEngine.Assertions;

namespace LlamaCpp
{
    public unsafe static class Common
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
            public float dry_multiplier;
            public float dry_base;
            public int dry_allowed_length;
            public int dry_penalty_last_n;

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
                    dry_multiplier = 0.0f,
                    dry_base = 1.75f,
                    dry_allowed_length = 2,
                    dry_penalty_last_n = -1,
                    top_n_sigma = -1.00f,
                };
            }
        }

        public static int[] common_tokenize(IntPtr vocab, string text, bool add_special, bool parse_special)
        {
            int n_tokens = text.Length + 2 * (add_special ? 1 : 0);
            int[] result = new int[n_tokens];
            n_tokens = Native.llama_tokenize(vocab, text, text.Length, result, result.Length, add_special, parse_special);
            if (n_tokens == int.MinValue)
            {
                throw new System.Exception("Tokenization failed: input text too large, tokenization result exceeds int limit");
            }
            if (n_tokens < 0)
            {
                Array.Resize(ref result, -n_tokens);
                int check = Native.llama_tokenize(vocab, text, text.Length, result, result.Length, add_special, parse_special);
                Assert.IsTrue(check == -n_tokens, "check tokens");
            }
            else
            {
                Array.Resize(ref result, n_tokens);
            }
            return result;
        }

        private static readonly Decoder decoder = Encoding.UTF8.GetDecoder();
        
        public static string common_token_to_piece(IntPtr vocab, int token, bool special)
        {
            byte[] piece = new byte[16];
            int n_chars = Native.llama_token_to_piece(vocab, token, piece, piece.Length, 0, special);
            if (n_chars < 0)
            {
                Array.Resize(ref piece, -n_chars);
                int check = Native.llama_token_to_piece(vocab, token, piece, piece.Length, 0, special);
            }
            else
            {
                Array.Resize(ref piece, n_chars);
            }

            // handle emoji
            int charCount = decoder.GetCharCount(piece, 0, piece.Length);
            char[] chars = new char[charCount];
            decoder.GetChars(piece, 0, piece.Length, chars, 0);
            
            return new string(chars);
        }
    }

}
