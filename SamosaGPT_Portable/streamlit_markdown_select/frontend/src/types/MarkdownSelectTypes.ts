export interface CodeBlock {
  code: string;
  language?: string;
  color?: string;
}

export interface SelectButton {
  text: string;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'danger';
  disabled?: boolean;
  action?: string;
}

export interface SelectOption {
  value: string;
  label: string;
  code_block?: CodeBlock;
  buttons?: SelectButton[];
  disabled?: boolean;
}