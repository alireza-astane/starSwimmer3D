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
    sphere { m*<-0.49228585409824366,-0.5926426998809136,-0.4775354556425182>, 1 }        
    sphere {  m*<0.9268816401019181,0.3972962139990037,9.371754641392629>, 1 }
    sphere {  m*<8.294668838424721,0.11220396320674197,-5.198922787681303>, 1 }
    sphere {  m*<-6.601294355264279,6.635285336827379,-3.708115884499695>, 1}
    sphere { m*<-3.614286482492758,-7.3917504735019435,-1.9232950757805851>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9268816401019181,0.3972962139990037,9.371754641392629>, <-0.49228585409824366,-0.5926426998809136,-0.4775354556425182>, 0.5 }
    cylinder { m*<8.294668838424721,0.11220396320674197,-5.198922787681303>, <-0.49228585409824366,-0.5926426998809136,-0.4775354556425182>, 0.5}
    cylinder { m*<-6.601294355264279,6.635285336827379,-3.708115884499695>, <-0.49228585409824366,-0.5926426998809136,-0.4775354556425182>, 0.5 }
    cylinder {  m*<-3.614286482492758,-7.3917504735019435,-1.9232950757805851>, <-0.49228585409824366,-0.5926426998809136,-0.4775354556425182>, 0.5}

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
    sphere { m*<-0.49228585409824366,-0.5926426998809136,-0.4775354556425182>, 1 }        
    sphere {  m*<0.9268816401019181,0.3972962139990037,9.371754641392629>, 1 }
    sphere {  m*<8.294668838424721,0.11220396320674197,-5.198922787681303>, 1 }
    sphere {  m*<-6.601294355264279,6.635285336827379,-3.708115884499695>, 1}
    sphere { m*<-3.614286482492758,-7.3917504735019435,-1.9232950757805851>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9268816401019181,0.3972962139990037,9.371754641392629>, <-0.49228585409824366,-0.5926426998809136,-0.4775354556425182>, 0.5 }
    cylinder { m*<8.294668838424721,0.11220396320674197,-5.198922787681303>, <-0.49228585409824366,-0.5926426998809136,-0.4775354556425182>, 0.5}
    cylinder { m*<-6.601294355264279,6.635285336827379,-3.708115884499695>, <-0.49228585409824366,-0.5926426998809136,-0.4775354556425182>, 0.5 }
    cylinder {  m*<-3.614286482492758,-7.3917504735019435,-1.9232950757805851>, <-0.49228585409824366,-0.5926426998809136,-0.4775354556425182>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    