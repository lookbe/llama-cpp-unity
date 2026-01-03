using UnityEngine;

namespace LlamaCpp
{
    public static class Backend
    {
        static int count = 0;

        public static void Init()
        {
            if (count == 0)
            {
                Native.llama_backend_init();
            }
            count++;
        }

        public static void Free()
        {
            count--;
            if (count == 0)
            {
                Native.llama_backend_free();
            }
        }
    }
}
