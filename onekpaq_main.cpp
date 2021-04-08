/* Copyright (C) Teemu Suutari */

#include <fstream>
#include <cstring>
#include <sys/stat.h>

#include "onekpaq_common.h"

#include "BlockCodec.hpp"
#include "StreamCodec.hpp"
#include "AsmDecode.hpp"

// references used to implement this:
// http://en.wikipedia.org/wiki/PAQ
// http://en.wikipedia.org/wiki/Context_mixing
// http://code4k.blogspot.com/2010/12/crinkler-secrets-4k-intro-executable.html
// clean implementation. no foreign code used.
// compared against paq8i and paq8l.

#define VERIFY_STREAM 1

// sans error handling
std::vector<u8> readFile(const std::string &fileName)
{
	std::vector<u8> ret;
	std::ifstream file(fileName.c_str(),std::ios::in|std::ios::binary|std::ios::ate);
	if (file.is_open()) {
		size_t fileLength=size_t(file.tellg());
		file.seekg(0,std::ios::beg);
		ret.resize(fileLength);
		file.read(reinterpret_cast<char*>(ret.data()),fileLength);
		file.close();
	}
	return ret;
}

// ditto
void writeFile(const std::string &fileName,const std::vector<u8> &src) {
	std::ofstream file(fileName.c_str(),std::ios::out|std::ios::binary|std::ios::trunc);
	if (file.is_open()) {
		file.write(reinterpret_cast<const char*>(src.data()),src.size());
		file.close();
	}
}

int main(int argc,char **argv)
{
	if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
		fprintf(stderr,
"Usage: %s <mode> <complexity> <input...> <output>\n"
"\n"
"\tmode: possible options:\n"
"\t\t1: single block (small decoder)\n"
"\t\t2: multi-block (small decoder)\n"
"\t\t3: single block (fast decoder)\n"
"\t\t4: multi-block (fast decoder)\n"
"\tcomplexity: ranges from 1 (low) to 3 (high). 'high' compresses smaller\n"
"\t            but takes a longer time to compress and decompress\n"
			, argv[0]);
		return 0;
	}

/*#ifdef __x86_64__
	INFO("oneKpaq v" ONEKPAQ_VERSION " 64-bit");
#else
	INFO("oneKpaq v" ONEKPAQ_VERSION " 32-bit");
#endif*/
	// Check for decode mode.
	if (std::string(argv[0]).find("onekpaq_decode")!=std::string::npos)
	{
		if (argc!=5) ABORT("usage: onekpaq_decode mode shift input.onekpaq output");

		StreamCodec::EncodeMode encodeMode = StreamCodec::EncodeMode(atoi(argv[1]));
		uint shift = atoi(argv[2]);

		struct stat st;
		//ASSERT(::stat(argv[4],&st)==-1,"Destination file exists");

		auto src=readFile(std::string(argv[3]));

		StreamCodec s2;
#if 1
		s2.AssignStream(encodeMode, shift, src);
#else
		s2.LoadStream(src);
#endif
		auto dest=s2.Decode();

		writeFile(std::string(argv[4]),dest);
	}
	// Simple encode (to stream that's readable with onekpaq_decode.
	else if (std::string(argv[0]).find("onekpaq_encode")!=std::string::npos) {
		if (argc<5) {
			ABORT("usage: onekpaq_encode mode complexity block1 block2 ... blockn output.onekpaq");
		}

		StreamCodec::EncodeMode mode=StreamCodec::EncodeMode(atoi(argv[1]));
		StreamCodec::EncoderComplexity complexity=StreamCodec::EncoderComplexity(atoi(argv[2]));

		std::vector<std::vector<u8>> blocks(argc-4);
		for (uint i=0;i<uint(argc-4);i++) {
			blocks[i]=readFile(std::string(argv[i+3]));
			ASSERT(blocks[i].size(),"Empty/missing file");
		}

		struct stat st;
		//ASSERT(::stat(argv[argc-1],&st)==-1,"Destination file exists");

		StreamCodec s;
		s.Encode(blocks,mode,complexity,"onekpaq_context.cache");
		auto stream=s.CreateSingleStream();

#ifdef VERIFY_STREAM
		INFO("Verifying...");

		StreamCodec s2;
		s2.LoadStream(stream);
		auto verify=s2.Decode();

		std::vector<u8> src;
		for (auto &it:blocks)
			src.insert(src.end(),it.begin(),it.end());

		ASSERT(src.size()==verify.size(),"size mismatch");

		for (uint i=0;i<src.size();i++)
			ASSERT(src[i]==verify[i],"%u: %02x!=%02x",i,src[i],verify[i]);
		INFO("Data verified");


		INFO("Verifying ASM stream...");
		auto verify2=s.DecodeAsmStream();
		for (uint i=0;i<src.size();i++)
			ASSERT(src[i]==verify2[i],"%u: %02x!=%02x",i,src[i],verify2[i]);
		INFO("ASM Data verified");
#endif

		fprintf(stdout/* not stderr */, "M mode=%u shift=%u\n", static_cast<unsigned>(s.getMode()), s.GetShift());
		std::vector<u8> outStream;
		outStream.insert(outStream.end(), stream.begin() + 10, stream.end());
		writeFile(std::string(argv[argc-1]), outStream);

	}
	// Default mode is encode to asm.
	else {
		if (argc<5) {
			ABORT("usage: onekpaq_encode mode complexity block1 block2 ... blockn output.onekpaq");
		}

		StreamCodec::EncodeMode mode=StreamCodec::EncodeMode(atoi(argv[1]));
		StreamCodec::EncoderComplexity complexity=StreamCodec::EncoderComplexity(atoi(argv[2]));

		std::vector<std::vector<u8>> blocks(argc-4);
		for (uint i=0;i<uint(argc-4);i++) {
			blocks[i]=readFile(std::string(argv[i+3]));
			ASSERT(blocks[i].size(),"Empty/missing file");
		}

		struct stat st;
		//ASSERT(::stat(argv[argc-1],&st)==-1,"Destination file exists");

		StreamCodec s;
		s.Encode(blocks,mode,complexity,"onekpaq_context.cache");
		auto stream=s.CreateSingleStream();

#ifdef VERIFY_STREAM
		INFO("Verifying...");

		StreamCodec s2;
		s2.LoadStream(stream);
		auto verify=s2.Decode();

		std::vector<u8> src;
		for (auto &it:blocks)
			src.insert(src.end(),it.begin(),it.end());

		ASSERT(src.size()==verify.size(),"size mismatch");

		for (uint i=0;i<src.size();i++)
			ASSERT(src[i]==verify[i],"%u: %02x!=%02x",i,src[i],verify[i]);
		INFO("Data verified");


		INFO("Verifying ASM stream...");
		auto verify2=s.DecodeAsmStream();
		for (uint i=0;i<src.size();i++)
			ASSERT(src[i]==verify2[i],"%u: %02x!=%02x",i,src[i],verify2[i]);
		INFO("ASM Data verified");

#ifndef __x86_64__
		INFO("Using actual ASM-decompressor");
		// now same thing with asm-decoder
		auto asmVerify=AsmDecode(s.GetAsmDest1(),s.GetAsmDest2(),mode,s.GetShift());

		// asm decoder does not know anything about size, we can only verify contents
		for (uint i=0;i<src.size();i++)
			ASSERT(src[i]==asmVerify[i],"%u: %02x!=%02x",i,src[i],asmVerify[i]);
#endif
		INFO("ASM-decompressor finished");
#endif

		auto src1 = s.GetAsmDest1();
		auto src2 = s.GetAsmDest2();
		std::vector<u8> combine=src1;
		combine.insert(combine.end(),src2.begin(),src2.end());
		for (int i=0;i<4;i++) combine.push_back(0);

		fprintf(stdout/* not stderr */, "P offset=%zu shift=%u\n", src1.size(), s.GetShift());

		writeFile(std::string(argv[argc-1]),combine/*stream*/);
        }

	return 0;
}

