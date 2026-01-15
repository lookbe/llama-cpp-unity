using System;
using System.Text;
using UnityEngine.Assertions;

namespace LlamaCpp
{
    public unsafe static class Common
    {
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
