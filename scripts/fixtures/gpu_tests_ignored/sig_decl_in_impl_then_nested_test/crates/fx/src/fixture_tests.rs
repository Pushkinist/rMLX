mod tests {
    trait L {
        fn load<T>(&self, x: T) -> u32
        where
            T: Copy;
    }

    #[test]
    fn gpu_i() {
        let device = Device::Gpu;
        if true {
            run(device);
        }
    }
}
