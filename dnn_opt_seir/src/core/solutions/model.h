class seir
{
public:

  void run(int time, int* history)
  {
    m_s = m_s_0;
    m_e = 0;
    m_i = m_i_0;
    m_j = m_j_0;
    m_r = 0;

    for(int i = 0; i < time; i++)
    {
      history[i * 5 + 0] = m_s;
      history[i * 5 + 1] = m_e;
      history[i * 5 + 2] = m_i;
      history[i * 5 + 3] = m_j;
      history[i * 5 + 4] = m_r;

      m_s += ds();
      m_e += de();
      m_i += di();
      m_j += dj();
      m_r += d_r();
    }
  } 

private:

  int n() const
  {
    return m_s + m_i + m_j + m_r;
  }

  float ds() const
  {
    int n = n();

    return -1.0f * m_beta * (m_i + m_tau * m_j) * m_s / n;
  }

  float de() const
  {
    int n = n();

    return m_beta * (m_i + m_tau * m_j) * m_s / n - 
           m_gamma_0 * m_e;
  }

  float di() const
  {
    return  m_gamma_0 * m_e * (1 - m_p_h)  - m_gamma_1 * m_i;
  }

  float dj() const
  {
    return  m_gamma_0 * m_e * m_p_h - m_gamma_1 * m_j;
  }

  float dr() const
  {
    return  m_gamma_1 * (m_i  + m_j);
  }

  int m_s;
  int m_e;
  int m_i;
  int m_j;
  int m_r;

  int m_s_0;
  int m_i_0;
  int m_j_0;

  float m_beta;
  float m_gamma_0;
  float m_gamma_1;
  float m_tau;
  float m_p_h;
}
