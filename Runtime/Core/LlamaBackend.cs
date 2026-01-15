using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;

namespace LlamaCpp
{
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
    public static class NativeDllPath
    {
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern IntPtr GetModuleHandle(string lpModuleName);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern uint GetModuleFileName(IntPtr hModule, StringBuilder lpFilename, int nSize);

        public static string GetDirectory(string dllName)
        {
            IntPtr hModule = GetModuleHandle(dllName);
            if (hModule == IntPtr.Zero)
                return null;

            var sb = new StringBuilder(1024);
            GetModuleFileName(hModule, sb, sb.Capacity);

            return Path.GetDirectoryName(sb.ToString());
        }
    }
#endif

    public static class Backend
    {
        static int count = 0;

        public static void EnableLog()
        {
            Native.llama_log_set(NativeLogCallback, IntPtr.Zero);
        }

        public static void Init()
        {
            if (count == 0)
            {
                Native.llama_max_devices();

#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
                string packagePath = NativeDllPath.GetDirectory("llama.dll");
                Native.ggml_backend_load(Path.Join(packagePath, "ggml-cpu-x64.dll"));
                Native.ggml_backend_load(Path.Join(packagePath, "ggml-vulkan.dll"));
#endif

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
                Native.llama_log_set(null, IntPtr.Zero);
            }
        }

        [AOT.MonoPInvokeCallback(typeof(Native.ggml_log_callback))]
        static void NativeLogCallback(int level, string text, IntPtr user_data)
        {
            if (level >= 4)
            {
                Debug.LogWarning("[Llama Native] " + text);
            }
            else
            {
                Debug.Log("[Llama Native] " + text);
            }
        }

    }
}
