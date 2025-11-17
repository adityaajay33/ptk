#ifndef RUNTIME_CORE_PORT_H_
#define RUNTIME_CORE_PORT_H_

namespace runtime {

    template <typename T>
    class OutputPort {
        public:

            OutputPort() : value_(nullptr) {}

            void Bind(T* value){
                value_ = value;
            }

            bool is_bound() const {
                return value_ != nullptr;
            }

            T* get() const {
                return value_;
            }

        private:
            T* value_;
    };

    template <typename T>
    class InputPort {
        public:

            InputPort() : value_(nullptr) {}

            void Bind(T* value){
                value_ = value;
            }

            bool is_bound() const {
                return value_ != nullptr;
            }

            const T* get() const {
                return value_;
            }

        private:
            const T* value_;
    };

}  // namespace runtime

#endif // RUNTIME_CORE_PORT_H_