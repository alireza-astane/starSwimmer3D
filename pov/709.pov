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
    sphere { m*<3.9033189174093954e-18,-5.4875029988392454e-18,0.8774925979600422>, 1 }        
    sphere {  m*<7.534116642455899e-19,-5.258881863287811e-18,5.907492597960066>, 1 }
    sphere {  m*<9.428090415820634,-6.341619615348256e-20,-2.45584073537329>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.45584073537329>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.45584073537329>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<7.534116642455899e-19,-5.258881863287811e-18,5.907492597960066>, <3.9033189174093954e-18,-5.4875029988392454e-18,0.8774925979600422>, 0.5 }
    cylinder { m*<9.428090415820634,-6.341619615348256e-20,-2.45584073537329>, <3.9033189174093954e-18,-5.4875029988392454e-18,0.8774925979600422>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.45584073537329>, <3.9033189174093954e-18,-5.4875029988392454e-18,0.8774925979600422>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.45584073537329>, <3.9033189174093954e-18,-5.4875029988392454e-18,0.8774925979600422>, 0.5}

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
    sphere { m*<3.9033189174093954e-18,-5.4875029988392454e-18,0.8774925979600422>, 1 }        
    sphere {  m*<7.534116642455899e-19,-5.258881863287811e-18,5.907492597960066>, 1 }
    sphere {  m*<9.428090415820634,-6.341619615348256e-20,-2.45584073537329>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.45584073537329>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.45584073537329>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<7.534116642455899e-19,-5.258881863287811e-18,5.907492597960066>, <3.9033189174093954e-18,-5.4875029988392454e-18,0.8774925979600422>, 0.5 }
    cylinder { m*<9.428090415820634,-6.341619615348256e-20,-2.45584073537329>, <3.9033189174093954e-18,-5.4875029988392454e-18,0.8774925979600422>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.45584073537329>, <3.9033189174093954e-18,-5.4875029988392454e-18,0.8774925979600422>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.45584073537329>, <3.9033189174093954e-18,-5.4875029988392454e-18,0.8774925979600422>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    