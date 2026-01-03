using System;
using System.Text;

namespace LlamaCpp
{
    public unsafe static class Chat
    {
        public class common_chat_msg
        {
            private byte[] roleBytes;
            private byte[] contentBytes;

            public string role
            {
                get
                {
                    if (roleBytes.Length == 0)
                        return "";

                    // Remove the last null byte before decoding
                    int len = roleBytes.Length;
                    if (roleBytes[len - 1] == 0)
                        len--;

                    return Encoding.UTF8.GetString(roleBytes, 0, len);
                }
                set
                {
                    if (string.IsNullOrEmpty(value))
                    {
                        roleBytes = new byte[] { 0 }; // just null terminator
                    }
                    else
                    {
                        var bytes = Encoding.UTF8.GetBytes(value);
                        roleBytes = new byte[bytes.Length + 1]; // +1 for null terminator
                        Array.Copy(bytes, roleBytes, bytes.Length);
                        roleBytes[roleBytes.Length - 1] = 0; // set null byte
                    }
                }
            }
            public string content
            {
                get
                {
                    if (contentBytes.Length == 0)
                        return "";

                    // Remove the last null byte before decoding
                    int len = contentBytes.Length;
                    if (contentBytes[len - 1] == 0)
                        len--;

                    return Encoding.UTF8.GetString(contentBytes, 0, len);
                }
                set
                {
                    if (string.IsNullOrEmpty(value))
                    {
                        contentBytes = new byte[] { 0 }; // just null terminator
                    }
                    else
                    {
                        var bytes = Encoding.UTF8.GetBytes(value);
                        contentBytes = new byte[bytes.Length + 1]; // +1 for null terminator
                        Array.Copy(bytes, contentBytes, bytes.Length);
                        contentBytes[contentBytes.Length - 1] = 0; // set null byte
                    }
                }
            }

            public byte[] roleUTF8 => roleBytes;
            public byte[] contentUTF8 => contentBytes;
        }

        public static string common_chat_format_single(IntPtr tmpl, common_chat_msg[] history, common_chat_msg input, bool user)
        {
            string formattedChat = string.Empty;
            if (history.Length > 0)
            {
                string formattedHistory = common_chat_templates_apply(tmpl, new common_chat_msg[] { history[history.Length - 1] }, user);
                if (formattedHistory.Length > 0 && formattedHistory[formattedHistory.Length - 1] == '\n')
                {
                    formattedChat = "\n";
                }
            }
            return formattedChat += common_chat_templates_apply(tmpl, new common_chat_msg[] { input }, user );
        }

        public static string common_chat_templates_apply(IntPtr tmpl, common_chat_msg[] inputs, bool user)
        {
            using Util.memory_pin_manager memory_manager = new Util.memory_pin_manager();
            {
                Native.llama_chat_message[] chat_msgs = new Native.llama_chat_message[inputs.Length];

                int buffer_len = 0;
                for (var i = 0; i < inputs.Length; i++)
                {
                    Memory<byte> roleMemory = inputs[i].roleUTF8.AsMemory();
                    Memory<byte> contentMemory = inputs[i].contentUTF8.AsMemory();

                    chat_msgs[i].role = memory_manager.Pin(roleMemory);
                    chat_msgs[i].content = memory_manager.Pin(contentMemory);

                    buffer_len += inputs[i].roleUTF8.Length;
                    buffer_len += inputs[i].contentUTF8.Length;
                }

                byte[] buffer = new byte[buffer_len];
                int res = Native.llama_chat_apply_template(tmpl, chat_msgs, chat_msgs.Length, user, buffer, buffer.Length);
                if (res < 0)
                {
                    return string.Empty;
                }

                Array.Resize<byte>(ref buffer, res);

                if (res > buffer_len)
                {
                    res = Native.llama_chat_apply_template(tmpl, chat_msgs, chat_msgs.Length, user, buffer, buffer.Length);
                }

                return Encoding.UTF8.GetString(buffer);
            }
        }
    }

}
