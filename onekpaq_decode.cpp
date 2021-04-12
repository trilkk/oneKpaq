#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <sys/time.h>

#include "onekpaq_common.h"

extern "C"
{

void DebugPrint(const char *str,...)
{
    va_list ap;
    va_start(ap,str);
    vfprintf(stderr,str,ap);
    fflush(stderr);
    va_end(ap);
}

void DebugPrintAndDie(const char *str,...)
{
    va_list ap;
    va_start(ap,str);
    vfprintf(stderr,str,ap);
    fflush(stderr);
    va_end(ap);
    abort();
}

}

#if defined(__APPLE__) || defined(HAS_LIBDISPATCH)

#include <dispatch/dispatch.h>

    template<typename F,typename... Args>
void DispatchLoop(size_t iterations,size_t stride,F func,Args&&... args)
{
    if (!iterations) return;
    if (stride<1) stride=1;
    dispatch_apply((iterations+stride-1)/stride,dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT,0),^(size_t majorIt) {
            majorIt*=stride;
            for (size_t minorIt=0;minorIt<stride;minorIt++) {
            size_t iter=majorIt+minorIt;
            if (iter>=iterations) break;
            func(iter,args...);
            }
            });
}

#else
#warning "Parallelization disabled!!!"

    template<typename F,typename... Args>
void DispatchLoop(size_t iterations,size_t stride,F func,Args&&... args)
{
    if (!iterations) return;
    // Take a coffee, come back, wait, go to lunch, come back, wait, take a coffee...
    for (size_t i=0;i<iterations;i++) func(i,args...);
}

#endif

class ArithDecoder
{
public:
    enum class DecoderType {
        SingleAsm=0,
        MultiAsm,
        Standard,
        TypeLast=Standard
    };

    ArithDecoder(const std::vector<u8> &src,DecoderType type);
    ~ArithDecoder() = default;

    u32 GetRange() const { return _range; }
    void ProcessEndOfSection(u32 subRange);
    void PreDecode(u32 subRange);
    bool Decode(u32 subRange);

private:
    void Normalize();

    DecoderType			_type;
    const std::vector<u8>&		_src;
    uint				_srcPos=0;
    uint				_destPos=0;
    u32				_value=0;
    u32				_range=1;
    bool				_hasPreBit=false;
};

class ArithEncoder
{
public:
    enum class EncoderType {
        SingleAsm=0,
        MultiAsm,
        Standard,
        TypeLast=Standard
    };

    ArithEncoder(EncoderType type);
    ~ArithEncoder() = default;

    u32 GetRange() const { return _range; }
    void PreEncode(u32 subRange);
    void Encode(bool bit,u32 subRange);

    void EndSection(u32 subRange);
    void Finalize();

    const std::vector<u8> &GetDest() const { return _dest; }

private:
    void Normalize();

    EncoderType		_type;
    std::vector<u8>		_dest;
    uint			_destPos=0;
    u32			_range=0x4000'0000U;
    u64			_value=0;
    uint			_bitCount=33;
    bool			_hasPreBit=false;
};

class BlockCodec
{
public:
    enum class BlockCodecType {
        Single=0,
        Standard,
        TypeLast=Standard
    };

    struct Probability;
    struct BitCounts;

private:
    struct Model
    {
        uint			model;
        uint			weight;
    };

    template<class CM,class CB,class MIX>
        class CompressorModel;

    class CompressorModelBase
    {
    public:
        CompressorModelBase() = default;
        virtual ~CompressorModelBase() = default;

        virtual uint GetModelCount() const = 0;
        virtual uint GetStartOffset() const = 0;

        virtual void CalculateAllProbabilities(std::vector<Probability> &dest,const std::vector<u8> &src,uint bitPos,uint shift) = 0;
        virtual float EstimateBitLength(const std::vector<Probability> &probabilities,const std::vector<Model> &models) = 0;

        virtual u32 CalculateSubRange(const std::vector<u8> &src,int bitPos,const std::vector<Model> &models,u32 range,uint shift) = 0;

    private:
        virtual void CreateWeightProfile(std::vector<std::pair<BitCounts,uint>> &dest,const std::vector<u8> &src,uint bitPos,const std::vector<Model> &models,uint shift) = 0;
    };

public:
    BlockCodec(BlockCodecType type,uint shift);
    ~BlockCodec() = default;

    const std::vector<u8> &GetHeader() const { return _header; }
    void SetHeader(const std::vector<u8> &header,uint size);

    void CreateContextModels(const std::vector<u8> &src,bool multipleRetryPoints,bool multipleInitPoints);
    void Encode(const std::vector<u8> &src,ArithEncoder &ec);
    uint GetRawLength() const { return _rawLength; }
    float GetEstimatedLength() const { return _estimatedLength; }
    std::string PrintModels() const;

    std::vector<u8> Decode(const std::vector<u8> &header,ArithDecoder &dc);

private:
    std::vector<u8> EncodeHeaderAndClean(std::vector<Model> &models);
    void DecodeHeader(const std::vector<u8> &header);
    void CreateCompressorModel();

    BlockCodecType			_type;
    std::vector<Model> 		_models;		// models, for encoding and decoding
    uint				_rawLength=0;		// unencoded data length

    std::vector<u8>			_header;		// for encoding only
    float				_estimatedLength=0;	// with header, encoding only

    uint				_shift;			// shift for encoding and decoding

    std::unique_ptr<CompressorModelBase> _cm;
};

class CacheFile {
public:
    CacheFile() { }
    ~CacheFile() = default;

    void clear(uint numBlocks);

    uint getNumBlocks() const { return _numBlocks; }
    void setShift(uint shift) { _shift=shift; }
    uint getShift() const { return _shift; }
    std::vector<uint> &getCombineData() { return _combine; }
    std::vector<std::vector<u8>> &getHeader() { return _header; }

    bool readFile(const std::string &fileName);
    bool writeFile(const std::string &fileName);

private:
    uint readUint(std::ifstream &stream);
    std::vector<u8> readVector(std::ifstream &stream);
    void writeUint(std::ofstream &stream,uint value);
    void writeVector(std::ofstream &stream,const std::vector<u8> &vector);

    uint				_shift=0;
    uint				_numBlocks=0;
    std::vector<uint>		_combine;
    std::vector<std::vector<u8>>	_header;
};

class StreamCodec
{
public:
    enum class EncodeMode : uint {
        Single=1,
        Multi,
        SingleFast,
        MultiFast,
        ModeLast=MultiFast
    };

    explicit StreamCodec() = default;

    ~StreamCodec() = default;

    EncodeMode getMode() const
    {
        return _mode;
    }
    uint GetShift() const
    {
        return _shift;
    }

    void AssignStream(EncodeMode encodeMode, uint shift, const std::vector<u8>& singleStream);
    std::vector<u8> Decode();

private:
    std::vector<std::vector<u8>> _header;
    std::vector<u8> _dest;		// encoded stream
    EncodeMode _mode=EncodeMode::Single;
    uint _shift;
};

template<typename F,typename... Args>
double Timer(F func,Args&&... args)
{
    struct timeval beforeTime;
    // should use something better that measures real CPU-time
    gettimeofday(&beforeTime,nullptr);
    func(args...);
    struct timeval afterTime;
    gettimeofday(&afterTime,nullptr);
    return afterTime.tv_sec-beforeTime.tv_sec+double(afterTime.tv_usec-beforeTime.tv_usec)/1000000.0;
}

ArithDecoder::ArithDecoder(const std::vector<u8> &src,DecoderType type) :
    _type(type),
    _src(src)
{
    ASSERT(_type<=DecoderType::TypeLast,"Unknown decoder type");
    Normalize();
}

void ArithDecoder::Normalize()
{
    auto decodeBit=[&]()->bool {
        bool ret=_srcPos>>3<_src.size()&&_src[_srcPos>>3]&0x80U>>(_srcPos&7);
        _srcPos++;
        return ret;
    };

    while (_range<0x4000'0000U) {
        _range<<=1;
        _value<<=1;
        if (_type!=DecoderType::Standard) {
            if (!_srcPos)  ASSERT(!decodeBit(),"Wrong start bit");
            if (_srcPos==6) ASSERT(decodeBit(),"Wrong anchor bit");
            if (_srcPos==7) ASSERT(!decodeBit(),"Wrong filler bit");
        }
        if (decodeBit()) _value++;
    }
}

void ArithDecoder::ProcessEndOfSection(u32 subRange)
{
    DEBUG("range %x, subrange %x, value %x",_range,subRange,_value);
    if (_type==DecoderType::SingleAsm) {
        ASSERT(subRange&&subRange<_range-1,"probability error");

        _range-=subRange;
        ASSERT(_value==_range,"End of section not detected");
        _range=1;
        _value=0;
        Normalize();
    }
    DEBUG("%u bits decoded",_srcPos);
}

void ArithDecoder::PreDecode(u32 subRange)
{
    if (_type!=DecoderType::Standard&&!_hasPreBit) {
        ASSERT(!Decode(subRange),"Wrong purge bit");
        _hasPreBit=true;
    }
}

bool ArithDecoder::Decode(u32 subRange)
{
    DEBUG("range %x, subrange %x, value %x",_range,subRange,_value);
    ASSERT(subRange&&subRange<_range-1,"probability error");

    bool ret;
    if (_type==DecoderType::SingleAsm) {
        _range-=subRange;
        ASSERT(_value!=_range,"End of section detected in middle of stream");
        ASSERT(!(_destPos&7)||_value!=_range+1,"Encoder encoded a symbol we bug on");
        if (_value>=_range) {
            _value-=_range+1;
            // _range=subRange-1 would be correct here.
            _range=subRange;
            ret=false;
        } else ret=true;
    } else {
        _range-=subRange;
        if (_value>=_range) {
            _value-=_range;
            _range=subRange;
            ret=false;
        } else ret=true;
    }

    Normalize();
    _destPos++;
    return ret;
}

ArithEncoder::ArithEncoder(EncoderType type) :
    _type(type)
{
    ASSERT(_type<=EncoderType::TypeLast,"Unknown encoder type");
}

void ArithEncoder::Normalize()
{
    auto encodeBit=[&](bool bit) {
        _dest.resize((_destPos>>3)+1);
        if (bit) _dest[_destPos>>3]|=0x80U>>(_destPos&7);
        _destPos++;
    };

    while (_range<0x4000'0000U) {
        if (_bitCount) _bitCount--;
        else {
            if (_type!=EncoderType::Standard) {
                if (!_destPos) encodeBit(false);	// start bit
                if (_destPos==6) encodeBit(true);	// anchor bit
                if (_destPos==7) encodeBit(false);	// filler bit
            }
            ASSERT(!(_value&0x8000'0000'0000'0000ULL),"Overflow");
            encodeBit(_value&0x4000'0000'0000'0000ULL);
            _value&=~0x4000'0000'0000'0000ULL;
        }
        _value<<=1;
        _range<<=1;
    }
}

void ArithEncoder::PreEncode(u32 subRange)
{
    if (_type!=EncoderType::Standard&&!_hasPreBit) {
        Encode(false,subRange);
        _hasPreBit=true;
    }
}

void ArithEncoder::Encode(bool bit,u32 subRange)
{
    DEBUG("bit %u, range %x, subrange %x, value %llx",bit,_range,subRange,_value);
    ASSERT(subRange&&subRange<_range-1,"probability error");

    if (_type==EncoderType::SingleAsm) {
        if (bit) _range-=subRange;
        else {
            _value+=_range-subRange+1;
            // _range=subRange-1 would be correct here.
            _range=subRange;
        }
    } else {
        if (bit) _range-=subRange;
        else {
            _value+=_range-subRange;
            _range=subRange;
        }
    }
    Normalize();
}

void ArithEncoder::EndSection(u32 subRange)
{
    DEBUG("range %x, subrange %x, value %llx",_range,subRange,_value);
    ASSERT(subRange&&subRange<_range-1,"probability error at the end");

    if (_type==EncoderType::SingleAsm) {
        _value+=_range-subRange;
        _range=1;
        Normalize();
    }
}

void ArithEncoder::Finalize()
{
#if 1
    for (uint i=0;i<63;i++) {
        _range>>=1;
        Normalize();
    }
#else
    u64 valueX=_value+_range^_value;
    while (!(valueX&0x4000'0000'0000'0000ULL)) {
        _range>>=1;
        Normalize();
        valueX<<=1;
    }
    _value|=0x4000'0000'0000'0000ULL;
    _range>>=1;
    Normalize();
#endif
    DEBUG("%u bits encoded",_destPos);
}

// intentionally not typedeffed float to disable math (needs mixer to do it properly)
struct BlockCodec::Probability
{
    float			p;
};

struct BlockCodec::BitCounts
{
    u32			c0;
    u32			c1;
};

static inline bool BitAtPos(const std::vector<u8> &src,uint bitPos)
{
    return src[bitPos>>3]&0x80U>>(bitPos&7);
}

class LogisticMixer
{
public:
    LogisticMixer() { }
    ~LogisticMixer() = default;

    static BlockCodec::Probability Convert(const BlockCodec::BitCounts &counts)
    {
        // squashed probability
        return BlockCodec::Probability{log2f((float)counts.c1/(float)counts.c0)};
    }

    void Add(BlockCodec::Probability p,uint weight)
    {
        _p+=p.p/weight;
    }

    float GetLength() const
    {
        // stretch + probability to length
        return -log2f(1/(exp2f(_p)+1));
    }

private:
    float 			_p=0;
};

class PAQ1CountBooster
{
public:
    PAQ1CountBooster() = delete;
    ~PAQ1CountBooster() = delete;

    static void Initialize(BlockCodec::BitCounts &counts)
    {
        counts.c0=0;
        counts.c1=0;
    }

    static void Increment(BlockCodec::BitCounts &counts,bool bitIsSet)
    {
        counts.c0++;
        counts.c1++;
        if (bitIsSet) counts.c0>>=1;
        else counts.c1>>=1;
    }

    static void Swap(BlockCodec::BitCounts &counts)
    {
        std::swap(counts.c0,counts.c1);
    }

    static void Finalize(BlockCodec::BitCounts &counts,uint shift)
    {
        counts.c0=(counts.c0<<shift)+1;
        counts.c1=(counts.c1<<shift)+1;
    }
};

class QWContextModel
{
public:
    class iterator
    {
        friend class QWContextModel;
    public:
        ~iterator() = default;

        iterator& operator++()
        {
            if (_currentPos!=_parent._bitPos+8>>3) {
                _currentPos++;
                findNext();
            }
            return *this;
        }

        bool operator!=(const iterator &it)
        {
            return it._currentPos!=_currentPos;
        }

        std::pair<uint,bool> operator*()
        {
            // returns negative model mask + bit
            return _maskBit;
        }

    private:
        iterator(QWContextModel &parent,uint currentPos) :
            _parent(parent),
            _currentPos(currentPos)
        {

        }

        iterator& findNext()
        {
            const std::vector<u8> &data=_parent._data;
            uint bitPos=_parent._bitPos;
            uint bytePos=bitPos>>3;
            uint maxPos=bitPos+8>>3;
            bool overRunBit=bytePos==_parent._data.size();
            u8 mask=0xff00U>>(bitPos&7);

            while (_currentPos<maxPos) {
                if (_currentPos>=8&&_currentPos<bytePos&&(overRunBit||!((data[_currentPos]^data[bytePos])&mask))) {
                    uint noMatchMask=0;
                    for (uint i=1;i<9;i++)
                        if (data[_currentPos-i]!=data[bytePos-i]) noMatchMask|=0x80>>i-1;

                    _maskBit=std::make_pair(noMatchMask,BitAtPos(data,(_currentPos<<3)+(bitPos&7)));
                    break;
                }
                _currentPos++;
            }
            return *this;
        }

        QWContextModel&	_parent;
        uint		_currentPos;
        std::pair<uint,bool> _maskBit;
    };

    QWContextModel(const std::vector<u8> &data,uint bitPos) :
        _data(data),
        _bitPos(bitPos)
    {

    }

    ~QWContextModel() = default;

    static uint GetModelCount()
    {
        return 0xffU;
    }

    static uint GetStartOffset()
    {
        return 9;
    }

    iterator begin()
    {
        return iterator(*this,0).findNext();
    }

    iterator end()
    {
        return iterator(*this,_bitPos+8>>3);
    }

private:
    const std::vector<u8>&	_data;
    uint			_bitPos;
};

// TODO: too much like previous, combine
class NoLimitQWContextModel
{
public:
    class iterator
    {
        friend class NoLimitQWContextModel;
    public:
        ~iterator() = default;

        iterator& operator++()
        {
            if (_currentPos!=_parent._bitPos+8>>3) {
                _currentPos++;
                findNext();
            }
            return *this;
        }

        bool operator!=(const iterator &it)
        {
            return it._currentPos!=_currentPos;
        }

        std::pair<uint,bool> operator*()
        {
            // returns negative model mask + bit
            return _maskBit;
        }

    private:
        iterator(NoLimitQWContextModel &parent,uint currentPos) :
            _parent(parent),
            _currentPos(currentPos)
        {

        }

        iterator& findNext()
        {
            const std::vector<u8> &data=_parent._data;
            uint bitPos=_parent._bitPos;
            uint bytePos=bitPos>>3;
            uint maxPos=bitPos+8>>3;
            bool overRunBit=bytePos==_parent._data.size();
            u8 mask=0xff00U>>(bitPos&7);

            while (_currentPos<maxPos) {
                if (bytePos<9&&!_currentPos) {
                    u8 xShift=8-(bitPos&7);
                    auto byteLookup=[&](uint pos)->u8 {
                        if (pos>bytePos||bytePos==_parent._data.size()) return 0;
                        else if (pos==bytePos) return data[pos]>>xShift;
                        else return data[pos];
                    };
                    auto rol=[&](u8 byte,u8 count)->u8 {
                        count&=7;
                        return (byte>>8-count)|(byte<<count);
                    };
                    u8 xByte=byteLookup(8)^rol(byteLookup(bytePos),xShift);
                    if (!(xByte>>xShift)) {
                        uint noMatchMask=0;
                        for (uint i=1;i<9;i++)
                            if (byteLookup(8-i)!=byteLookup(bytePos-i)) noMatchMask|=0x80>>i-1;
                        _maskBit=std::make_pair(noMatchMask,xByte>>xShift-1&1);
                        break;
                    }
                } else {
                    if (_currentPos>=8&&_currentPos<bytePos&&(overRunBit||!((data[_currentPos]^data[bytePos])&mask))) {
                        uint noMatchMask=0;
                        for (uint i=1;i<9;i++)
                            if (data[_currentPos-i]!=data[bytePos-i]) noMatchMask|=0x80>>i-1;

                        _maskBit=std::make_pair(noMatchMask,BitAtPos(data,(_currentPos<<3)+(bitPos&7)));
                        break;
                    }

                }
                _currentPos++;
            }
            return *this;
        }

        NoLimitQWContextModel&	_parent;
        uint			_currentPos;
        std::pair<uint,bool>	_maskBit;
    };

    NoLimitQWContextModel(const std::vector<u8> &data,uint bitPos) :
        _data(data),
        _bitPos(bitPos)
    {

    }

    ~NoLimitQWContextModel() = default;

    static uint GetModelCount()
    {
        return 0xffU;
    }

    static uint GetStartOffset()
    {
        return 0;
    }

    iterator begin()
    {
        return iterator(*this,0).findNext();
    }

    iterator end()
    {
        return iterator(*this,_bitPos+8>>3);
    }

private:
    const std::vector<u8>&	_data;
    uint			_bitPos;
};

template<class CM,class CB,class MIX>
class BlockCodec::CompressorModel : public BlockCodec::CompressorModelBase
{
public:
    CompressorModel() { }

    ~CompressorModel() = default;

    virtual uint GetModelCount() const override
    {
        return CM::GetModelCount();
    }

    virtual uint GetStartOffset() const override
    {
        return CM::GetStartOffset();
    }

    virtual void CalculateAllProbabilities(std::vector<Probability> &dest,const std::vector<u8> &src,uint bitPos,uint shift) override
    {
        CM cm(src,bitPos);

        BitCounts counts[CM::GetModelCount()];
        for (auto &it:counts) CB::Initialize(it);
        for (auto m:cm) {
            uint i=0;
            for (auto &it:counts) {
                if (!(m.first&i)) CB::Increment(it,m.second);
                i++;
            }
        }
        dest.clear();
        for (auto &it:counts) {
            if (BitAtPos(src,bitPos)) CB::Swap(it);		// simplifies processing, Probabilities will be invariant from src
            CB::Finalize(it,shift);
            dest.push_back(MIX::Convert(it));
        }
    }

    virtual float EstimateBitLength(const std::vector<Probability> &probabilities,const std::vector<BlockCodec::Model> &models) override
    {
        MIX mixer;

        for (auto &it:models)
            if (it.weight) mixer.Add(probabilities[it.model],it.weight);

        return mixer.GetLength();
    }

    virtual u32 CalculateSubRange(const std::vector<u8> &src,int bitPos,const std::vector<BlockCodec::Model> &models,u32 range,uint shift) override;

    virtual void CreateWeightProfile(std::vector<std::pair<BitCounts,uint>> &dest,const std::vector<u8> &src,uint bitPos,const std::vector<BlockCodec::Model> &models,uint shift) override
    {
        CM cm(src,bitPos);

        dest.clear();
        BitCounts counts[models.size()];
        for (auto &it:counts) CB::Initialize(it);
        for (auto m:cm) {
            uint i=0;
            for (auto &it:models) {
                if (!(m.first&it.model)) CB::Increment(counts[i],m.second);
                i++;
            }
        }
        uint i=0;
        for (auto &it:models) {
            CB::Finalize(counts[i],shift);
            dest.push_back(std::make_pair(counts[i++],it.weight));
        }
    }
};

    template<>
u32 BlockCodec::CompressorModel<QWContextModel,PAQ1CountBooster,LogisticMixer>::CalculateSubRange(const std::vector<u8> &src,int bitPos,const std::vector<BlockCodec::Model> &models,u32 range,uint shift)
{
    // implementation is correct. however, there might be rounding errors
    // ruining the result.
    // please note the funny looking type-conversions here. they are for a reason
    ASSERT(bitPos>=-1,"Invalid pos");

    std::vector<std::pair<BitCounts,uint>> def;
    if (bitPos==-1) {
        // for the pre-bit no models says a thing
        BitCounts bc;
        PAQ1CountBooster::Initialize(bc);
        PAQ1CountBooster::Finalize(bc,shift);
        for (auto &it:models)
            def.push_back(std::make_pair(bc,it.weight));
    } else {
        CreateWeightProfile(def,src,bitPos,models,shift);
    }

    long double p=1;
    uint weight=def[0].second;
    for (auto &it:def) {
        while (weight!=it.second) {
            asm volatile("fsqrt\n":"=t"(p):"0"(p));
            weight>>=1;
        }
        p=(long double)(int)it.first.c0/p;
        p=(long double)(int)it.first.c1/p;
    }
    while (weight!=1) {
        asm volatile("fsqrt\n":"=t"(p):"0"(p));
        weight>>=1;
    }
    u32 ret;
    asm volatile("fistl %0\n":"=m"(ret):"t"((long double)(int)range / (1+p)));
    return ret;
}

// TODO: remove cut-copy-paste
    template<>
u32 BlockCodec::CompressorModel<NoLimitQWContextModel,PAQ1CountBooster,LogisticMixer>::CalculateSubRange(const std::vector<u8> &src,int bitPos,const std::vector<BlockCodec::Model> &models,u32 range,uint shift)
{
    // ditto
    ASSERT(bitPos>=-1,"Invalid pos");

    std::vector<std::pair<BitCounts,uint>> def;
    if (bitPos==-1) {
        BitCounts bc;
        // All the models have a hit for a zero bit (which it really is!)
        PAQ1CountBooster::Initialize(bc);
        PAQ1CountBooster::Increment(bc,false);
        PAQ1CountBooster::Finalize(bc,shift);
        for (auto &it:models)
            def.push_back(std::make_pair(bc,it.weight));
    } else {
        CreateWeightProfile(def,src,bitPos,models,shift);
    }

    long double p=1;
    uint weight=def[0].second;
    for (auto &it:def) {
        while (weight!=it.second) {
            asm volatile("fsqrt\n":"=t"(p):"0"(p));
            weight>>=1;
        }
        p=(long double)(int)it.first.c0/p;
        p=(long double)(int)it.first.c1/p;
    }
    while (weight!=1) {
        asm volatile("fsqrt\n":"=t"(p):"0"(p));
        weight>>=1;
    }
    u32 ret;
    asm volatile("fistl %0\n":"=m"(ret):"t"((long double)(int)range / (1+p)));
    return ret;
}

// ---

BlockCodec::BlockCodec(BlockCodecType type,uint shift) :
    _type(type),
    _shift(shift)
{
    ASSERT(_type<=BlockCodecType::TypeLast,"Unknown type");
}

void BlockCodec::SetHeader(const std::vector<u8> &header,uint size)
{
    DecodeHeader(header);
    _rawLength=size;
    _header=EncodeHeaderAndClean(_models);
}

std::vector<u8> BlockCodec::EncodeHeaderAndClean(std::vector<Model> &models)
{
    std::vector<u8> header;

    header.clear();
    ASSERT(_rawLength<65536,"Too long buffer");
    header.push_back(_rawLength);
    header.push_back(_rawLength>>8);
    header.push_back(0);						// empty for now

    // clean up first!
    std::sort(models.begin(),models.end(),[&](const Model &a,const Model &b) {
            if (a.weight>b.weight) return true;
            if (a.weight<b.weight) return false;
            ASSERT(a.model!=b.model,"duplicate model");
            return a.model>b.model;
            });
    ASSERT(models.size(),"empty models");
    while (!models.back().weight) models.pop_back();

    uint weight=models[0].weight;
    uint model=0x100;
    for (auto &it:models) {
        if (weight!=it.weight && model<it.model) weight>>=1;	// free double
        while (weight!=it.weight) {
            header.push_back(model);
            weight>>=1;
        }
        model=it.model;
        header.push_back(model);
    }
    while (weight!=1) {
        header.push_back(model);
        weight>>=1;
    }
    ASSERT(header.size()<=123,"Too long header");
    header[2]=header.size();
    return header;
}

void BlockCodec::DecodeHeader(const std::vector<u8> &header)
{
    ASSERT(header.size()>=4,"Too short header");
    uint size=header[0]+(uint(header[1])<<8);
    ASSERT(size,"null size");
    _rawLength=size;
    _models.clear();

    uint weight=1;
    uint model=0x100;
    for (uint i=3;i<header[2];i++) {
        if (header[i]>=model) {
            for (auto &m:_models) m.weight<<=1;
            if (header[i]==model) continue;
        }
        _models.push_back(Model{header[i],weight});
        model=header[i];
    }
}

void BlockCodec::CreateContextModels(const std::vector<u8> &src,bool multipleRetryPoints,bool multipleInitPoints)
{
    // TODO: this code is messy after few rounds of refactoring...
    const uint numWeights=7;	// try from 1/2^2 to 1/2^8
    uint retryCount=multipleRetryPoints?4:1;

    CreateCompressorModel();

    // too short block is non estimatable block
    if (src.size()<=_cm->GetStartOffset()) {
        _models.clear();
        _models.push_back(Model{0,2});
        _rawLength=uint(src.size());
        _header=EncodeHeaderAndClean(_models);
        _estimatedLength=uint(_header.size()+src.size())*8;
        return;
    }

    // first bytes do not have any modeling and thus can be excluded here
    uint bitLength=uint(src.size()-_cm->GetStartOffset())*8;
    std::vector<std::vector<Probability>> probabilityMap(bitLength);

    TICK();
    DispatchLoop(bitLength,32,[&](size_t i) {
            _cm->CalculateAllProbabilities(probabilityMap[i],src,uint(i+_cm->GetStartOffset()*8),_shift);
            });


    auto StaticLength=[&](std::vector<Model> &models)->float {
        return _cm->GetStartOffset()*8;
    };

    auto IterateModels=[&](std::vector<Model> &models)->std::pair<std::vector<Model>,float> {
        // parallelization makes this a tad more complicated.
        // first loop and then decide the best from the candidates
        size_t modelListSize=_cm->GetModelCount()*(numWeights+models.size());
        std::vector<std::pair<std::vector<Model>,float>> modelList(modelListSize);
        for (auto &it:modelList) it.second=std::numeric_limits<float>::infinity();

        DispatchLoop(modelListSize,1,[&](size_t combinedIndex) {
                uint i=uint(combinedIndex/_cm->GetModelCount());
                uint newModel=uint(combinedIndex%_cm->GetModelCount());
                std::vector<Model> testModels=models;
                bool modified=false;
                auto current=std::find_if(testModels.begin(),testModels.end(),[&](const Model &a) {
                        return a.model==newModel;
                        });
                if (i<numWeights)
                {
                uint weight=2<<i;
                if (current==testModels.end()) {
                // test adding a model
                testModels.push_back(Model{newModel,weight});
                modified=true;
                } else if (testModels.size()!=1 || current->weight!=weight) {
                // test changing a weight
                // if weight is the same, test removing model
                if (current->weight==weight) current->weight=0;
                else current->weight=weight;
                modified=true;
                }
                } else {
                    if (current==testModels.end()) {
                        // try swapping a model
                        i-=numWeights;
                        if (testModels[i].model!=newModel) {
                            testModels[i].model=newModel;
                            modified=true;
                        }
                    }
                }
                if (modified) {
                    float length=0;
                    for (uint j=0;j<bitLength;j++)
                        length+=_cm->EstimateBitLength(probabilityMap[j],testModels);
                    length+=EncodeHeaderAndClean(testModels).size()*8;
                    modelList[combinedIndex].first=testModels;
                    modelList[combinedIndex].second=length;
                }
        });

        auto ret=std::min_element(modelList.begin(),modelList.end(),[&](const std::pair<std::vector<Model>,float> &a,const std::pair<std::vector<Model>,float> &b) { return a.second<b.second; });
        return *ret;
    };

    // Find ideal context set
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<uint> d(0,_cm->GetModelCount()-1);

    std::vector<std::vector<uint>> seedWeights{
        {},
            {2,2},
            {2,4,8,8},
            {4,4,4,4},
            {4,4,8,8,8,8},
            {4,8,8,8,8,16,16,16,16},
            {16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16}};

    std::vector<Model> currentBestModels;
    float currentBestLength=std::numeric_limits<float>::infinity();
    float staticLength=StaticLength(currentBestModels);

    for (auto &sw:seedWeights) {
        for (uint i=0;i<retryCount;i++) {
            std::vector<Model> testModels=currentBestModels;
            float testLength=std::numeric_limits<float>::infinity();
            for (auto w:sw) {
                uint model=d(mt);
                while (std::find_if(testModels.begin(),testModels.end(),[&](Model &m){return m.model==model;})!=testModels.end())
                    model=d(mt);
                testModels.push_back(Model{model,w});
            }
            for (;;) {
                auto best=IterateModels(testModels);
                TICK();
                best.second+=staticLength;
                if (best.second<testLength) {
                    testModels=best.first;
                    testLength=best.second;
                } else break;
            }
            if (testLength<currentBestLength) {
                currentBestModels=testModels;
                currentBestLength=testLength;
            }
            if (!sw.size()) break;
        }
        if (!multipleInitPoints) break;
    }

    // mark length
    _rawLength=uint(src.size());

    // will scale and sort
    _models=currentBestModels;
    _header=EncodeHeaderAndClean(_models);
    _estimatedLength=currentBestLength;
}

void BlockCodec::Encode(const std::vector<u8> &src,ArithEncoder &ec)
{
    uint bitLength=uint(src.size()*8);

    CreateCompressorModel();
    ec.PreEncode(_cm->CalculateSubRange(src,-1,_models,ec.GetRange(),_shift));
    for (uint i=0;i<bitLength;i++)
        ec.Encode(BitAtPos(src,i),_cm->CalculateSubRange(src,i,_models,ec.GetRange(),_shift));
    ec.EndSection(_cm->CalculateSubRange(src,bitLength,_models,ec.GetRange(),_shift));
}

std::vector<u8> BlockCodec::Decode(const std::vector<u8> &header,ArithDecoder &dc)
{
    DecodeHeader(header);

    std::vector<u8> ret(_rawLength);
    uint bitLength=uint(ret.size()*8);

    CreateCompressorModel();
    dc.PreDecode(_cm->CalculateSubRange(ret,-1,_models,dc.GetRange(),_shift));
    for (uint i=0;i<bitLength;i++)
        if (dc.Decode(_cm->CalculateSubRange(ret,i,_models,dc.GetRange(),_shift)))
            ret[i>>3]|=0x80U>>(i&7);
    dc.ProcessEndOfSection(_cm->CalculateSubRange(ret,bitLength,_models,dc.GetRange(),_shift));
    return ret;
}

void BlockCodec::CreateCompressorModel()
{
    switch (_type) {
    case BlockCodecType::Single:
        _cm.reset(new CompressorModel<NoLimitQWContextModel,PAQ1CountBooster,LogisticMixer>());
        break;

    case BlockCodecType::Standard:
        _cm.reset(new CompressorModel<QWContextModel,PAQ1CountBooster,LogisticMixer>());
        break;

    default:
        break;
    }
}

std::string BlockCodec::PrintModels() const
{
    std::string ret;
    {
        char tmp[32];
        sprintf(tmp,"Shift=%u ",_shift);
        ret+=tmp;
    }
    for (auto &it:_models) {
        char tmp[32];
        sprintf(tmp,"(1/%u)*0x%02x ",it.weight,it.model);
        ret+=tmp;
    }
    return ret;
}

static BlockCodec::BlockCodecType StreamModetoBlockCodecType(StreamCodec::EncodeMode em)
{
    return (BlockCodec::BlockCodecType[]){
        BlockCodec::BlockCodecType::Single,
            BlockCodec::BlockCodecType::Standard,
            BlockCodec::BlockCodecType::Single,
            BlockCodec::BlockCodecType::Standard
    }[static_cast<uint>(em)-1];
}

void CacheFile::clear(uint numBlocks)
{
    _numBlocks=numBlocks;
    _combine.clear();
    _header.clear();
}

bool CacheFile::readFile(const std::string &fileName)
{
    bool ret=false;
    std::ifstream file(fileName.c_str(),std::ios::in|std::ios::binary);
    if (file.is_open()) {
        _shift=readUint(file);
        _numBlocks=readUint(file);
        uint numCombinedBlocks=readUint(file);
        _combine.clear();
        _header.clear();
        for (uint i=0;file&&i<numCombinedBlocks;i++) {
            _combine.push_back(readUint(file));
            _header.push_back(readVector(file));
        }
        ret=bool(file);
        file.close();
    }
    return ret;
}

bool CacheFile::writeFile(const std::string &fileName)
{
    bool ret=false;
    std::ofstream file(fileName.c_str(),std::ios::out|std::ios::binary|std::ios::trunc);
    if (file.is_open()) {
        writeUint(file,_shift);
        writeUint(file,_numBlocks);
        writeUint(file,uint(_header.size()));
        for (uint i=0;file&&i<_header.size();i++) {
            writeUint(file,_combine[i]);
            writeVector(file,_header[i]);
        }
        ret=bool(file);
        file.close();
    }
    return ret;
}

uint CacheFile::readUint(std::ifstream &stream)
{
    if (!stream) return 0;

    // x86 only
    u32 ret=0;
    stream.read(reinterpret_cast<char*>(&ret),sizeof(ret));
    return ret;
}

std::vector<u8> CacheFile::readVector(std::ifstream &stream)
{
    std::vector<u8> ret;
    uint length=readUint(stream);
    if (stream&&length) {
        ret.resize(length);
        stream.read(reinterpret_cast<char*>(ret.data()),length);
    }
    return ret;
}

void CacheFile::writeUint(std::ofstream &stream,uint value)
{
    if (!stream) return;

    // x86 only
    u32 tmp=value;
    stream.write(reinterpret_cast<const char*>(&tmp),sizeof(tmp));
}

void CacheFile::writeVector(std::ofstream &stream,const std::vector<u8> &vector)
{
    writeUint(stream,uint(vector.size()));
    if (stream&&vector.size()) stream.write(reinterpret_cast<const char*>(vector.data()),vector.size());
}

void StreamCodec::AssignStream(EncodeMode encodeMode, uint shift, const std::vector<u8>& singleStream)
{
    _mode = encodeMode;
    ASSERT(uint(_mode) && _mode <= EncodeMode::ModeLast, "Unknown mode");
    _shift = shift;
    ASSERT(_shift && _shift <= 16, "Wrong shift");

    _header.clear();
    uint prevEnd=0;
    bool isSingle=_mode!=EncodeMode::Multi&&_mode!=EncodeMode::MultiFast;
    while (singleStream[prevEnd]||singleStream[prevEnd+1]) {
        uint headerEnd=singleStream[prevEnd+2]+prevEnd;
        _header.push_back(std::vector<u8>(singleStream.begin()+prevEnd,singleStream.begin()+headerEnd));
        prevEnd=headerEnd;
        if (isSingle) break;
    }
    //INFO("prevEnd=%u isSingle=%u", prevEnd, isSingle);
    _dest=std::vector<u8>(singleStream.begin()+prevEnd+(isSingle?0:2),singleStream.end());
}

std::vector<u8> StreamCodec::Decode()
{
    // now it is simple
    std::vector<u8> ret;
    auto timeTaken=Timer([&]() {
            ArithDecoder dc(_dest,ArithDecoder::DecoderType::Standard);
            for (auto &it:_header) {
            BlockCodec bc(StreamModetoBlockCodecType(_mode),_shift);
            auto block=bc.Decode(it,dc);
            ret.insert(ret.end(),block.begin(),block.end());
            }
            });
    INFO("Decoding stream took %f seconds",float(timeTaken));
    return ret;
}

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

int main(int argc, char** argv)
{
#if 1
    if (argc!=5) ABORT("usage: onekpaq_decode mode shift input.onekpaq output");

    StreamCodec::EncodeMode encodeMode = StreamCodec::EncodeMode(atoi(argv[1]));
    uint shift = atoi(argv[2]);

    auto src=readFile(std::string(argv[3]));

    StreamCodec s2;
    s2.AssignStream(encodeMode, shift, src);
    auto dest=s2.Decode();

    writeFile(std::string(argv[4]),dest);
#endif
    return 0;
}
