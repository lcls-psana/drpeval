import "regent"

local c = regentlib.c

struct SlacConfig
{
	parallelism : int32;
	copies      : int32;
}

local cstring = terralib.includec("string.h")

terra print_usage_and_abort()
  c.printf("Usage: regent.py slac.rg [OPTIONS]\n")
  c.printf("OPTIONS\n")
  c.printf("  -h            : Print the usage and exit.\n")
  c.abort()
end

terra file_exists(filename : rawstring)
  var file = c.fopen(filename, "rb")
  if file == nil then return false end
  c.fclose(file)
  return true
end

terra SlacConfig:initialize_from_command()
  var args = c.legion_runtime_get_input_args()
  var i = 1
  self.parallelism = 1
  self.copies = 32
  while i < args.argc do
    if cstring.strcmp(args.argv[i], "-h") == 0 then
      print_usage_and_abort()
		elseif cstring.strcmp(args.argv[i], "-p") == 0 then
      i = i + 1
      self.parallelism = c.atoi(args.argv[i])
		elseif cstring.strcmp(args.argv[i], "-c") == 0 then
      i = i + 1
      self.copies = c.atoi(args.argv[i])
    end
    i = i + 1
  end
end

return SlacConfig
