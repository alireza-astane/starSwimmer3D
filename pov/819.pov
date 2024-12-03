#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-1.8109297952144185e-18,-4.997569091217398e-18,1.002483728593136>, 1 }        
    sphere {  m*<-1.0293087874248913e-18,-4.325040421898956e-18,5.262483728593163>, 1 }
    sphere {  m*<9.428090415820634,2.9016299835932876e-20,-2.3308496047401963>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.3308496047401963>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.3308496047401963>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-1.0293087874248913e-18,-4.325040421898956e-18,5.262483728593163>, <-1.8109297952144185e-18,-4.997569091217398e-18,1.002483728593136>, 0.5 }
    cylinder { m*<9.428090415820634,2.9016299835932876e-20,-2.3308496047401963>, <-1.8109297952144185e-18,-4.997569091217398e-18,1.002483728593136>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.3308496047401963>, <-1.8109297952144185e-18,-4.997569091217398e-18,1.002483728593136>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.3308496047401963>, <-1.8109297952144185e-18,-4.997569091217398e-18,1.002483728593136>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-1.8109297952144185e-18,-4.997569091217398e-18,1.002483728593136>, 1 }        
    sphere {  m*<-1.0293087874248913e-18,-4.325040421898956e-18,5.262483728593163>, 1 }
    sphere {  m*<9.428090415820634,2.9016299835932876e-20,-2.3308496047401963>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.3308496047401963>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.3308496047401963>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-1.0293087874248913e-18,-4.325040421898956e-18,5.262483728593163>, <-1.8109297952144185e-18,-4.997569091217398e-18,1.002483728593136>, 0.5 }
    cylinder { m*<9.428090415820634,2.9016299835932876e-20,-2.3308496047401963>, <-1.8109297952144185e-18,-4.997569091217398e-18,1.002483728593136>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.3308496047401963>, <-1.8109297952144185e-18,-4.997569091217398e-18,1.002483728593136>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.3308496047401963>, <-1.8109297952144185e-18,-4.997569091217398e-18,1.002483728593136>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    