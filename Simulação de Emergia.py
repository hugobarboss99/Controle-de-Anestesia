import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Função para o modelo do paciente (simulado)
def patient_model(y, t, u, emergency=False):
    dydt = -0.5 * y + u
    if emergency and t > 5:
        dydt -= 2.0  # Simulação de uma emergência, como queda na pressão arterial
    return dydt

# Parâmetros do controlador PID
Kp = 2.0
Ki = 1.0
Kd = 0.5

# Sinais simulados
setpoint = 1.0
y = 0.0
integral = 0.0
previous_error = 0.0

# Arrays para armazenar os resultados
time = np.linspace(0, 10, 100)
yout = np.zeros_like(time)
uout = np.zeros_like(time)

# Simulação
for i in range(1, len(time)):
    dt = time[i] - time[i-1]
    error = setpoint - y
    integral += error * dt
    derivative = (error - previous_error) / dt
    u = Kp*error + Ki*integral + Kd*derivative
    
    # Armazenando dados
    uout[i] = u
    y = odeint(patient_model, y, [0, dt], args=(u, True))[-1]
    yout[i] = y
    previous_error = error

# Plot dos resultados
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, yout, 'b-', label='Sinal Vital (simulado)')
plt.plot(time, setpoint * np.ones_like(time), 'r--', label='Setpoint')
plt.xlabel('Tempo (s)')
plt.ylabel('Resposta')
plt.legend()
plt.title('Resposta do Controle PID com Emergência')

plt.subplot(2, 1, 2)
plt.plot(time, uout, 'g-', label='Ajuste de Anestesia')
plt.xlabel('Tempo (s)')
plt.ylabel('Controle')
plt.legend()
plt.title('Saída do Controle PID com Emergência')

plt.tight_layout()
plt.show()
