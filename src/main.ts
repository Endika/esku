import '@presentation/styles/global.css';
import { renderApp } from '@presentation/App';

const root = document.querySelector<HTMLElement>('#app');
if (!root) throw new Error('Missing #app mount point');

renderApp(root);
