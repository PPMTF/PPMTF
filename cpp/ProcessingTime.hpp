#include<iostream>
#include<string>
#include<chrono>

class  ProcessingTime{
    public:
        ProcessingTime(const std::string& name = "Process", bool start = true) :
            m_name(name),
            m_isActive(start)
        {
            if (start)
            {
                this->restart();
            }
        }
        ~ProcessingTime()
        {
            this->stop();
        }

        ///<summary>
        ///Restart measuring time
        ///</summary>
        void restart()&
        {
            m_start = std::chrono::system_clock::now();
            m_isActive = true;
        }
        ///<summary>
        ///Stop measuring time and output the results
        ///</summary>
        void stop()&
        {
            if (!m_isActive)
                return;
            const auto end = std::chrono::system_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds >(end - m_start).count();
            std::cout << elapsed <<"ms"<< std::endl;

            m_isActive = false;
        }
    private:

        std::string m_name;
        std::chrono::system_clock::time_point m_start;
        bool m_isActive;

};
