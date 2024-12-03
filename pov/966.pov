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
    sphere { m*<-1.0714661816837868e-18,-5.3939608858453695e-18,1.1615776394246755>, 1 }        
    sphere {  m*<-5.174634477376874e-18,-5.3763590665862615e-18,4.392577639424719>, 1 }
    sphere {  m*<9.428090415820634,1.3387691616497102e-19,-2.1717556939086573>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.1717556939086573>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.1717556939086573>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-5.174634477376874e-18,-5.3763590665862615e-18,4.392577639424719>, <-1.0714661816837868e-18,-5.3939608858453695e-18,1.1615776394246755>, 0.5 }
    cylinder { m*<9.428090415820634,1.3387691616497102e-19,-2.1717556939086573>, <-1.0714661816837868e-18,-5.3939608858453695e-18,1.1615776394246755>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.1717556939086573>, <-1.0714661816837868e-18,-5.3939608858453695e-18,1.1615776394246755>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.1717556939086573>, <-1.0714661816837868e-18,-5.3939608858453695e-18,1.1615776394246755>, 0.5}

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
    sphere { m*<-1.0714661816837868e-18,-5.3939608858453695e-18,1.1615776394246755>, 1 }        
    sphere {  m*<-5.174634477376874e-18,-5.3763590665862615e-18,4.392577639424719>, 1 }
    sphere {  m*<9.428090415820634,1.3387691616497102e-19,-2.1717556939086573>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.1717556939086573>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.1717556939086573>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-5.174634477376874e-18,-5.3763590665862615e-18,4.392577639424719>, <-1.0714661816837868e-18,-5.3939608858453695e-18,1.1615776394246755>, 0.5 }
    cylinder { m*<9.428090415820634,1.3387691616497102e-19,-2.1717556939086573>, <-1.0714661816837868e-18,-5.3939608858453695e-18,1.1615776394246755>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.1717556939086573>, <-1.0714661816837868e-18,-5.3939608858453695e-18,1.1615776394246755>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.1717556939086573>, <-1.0714661816837868e-18,-5.3939608858453695e-18,1.1615776394246755>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    