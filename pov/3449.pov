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
    sphere { m*<0.1831613272120411,0.5541186317434517,-0.021927940013526576>, 1 }        
    sphere {  m*<0.4238964319537327,0.6828287099237772,2.965626831107024>, 1 }
    sphere {  m*<2.9178697212182985,0.656152607129826,-1.2511374654647112>, 1 }
    sphere {  m*<-1.4384540326808493,2.882592576162053,-0.9958737054294969>, 1}
    sphere { m*<-2.908079589840338,-5.289429130322966,-1.8129734199254401>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4238964319537327,0.6828287099237772,2.965626831107024>, <0.1831613272120411,0.5541186317434517,-0.021927940013526576>, 0.5 }
    cylinder { m*<2.9178697212182985,0.656152607129826,-1.2511374654647112>, <0.1831613272120411,0.5541186317434517,-0.021927940013526576>, 0.5}
    cylinder { m*<-1.4384540326808493,2.882592576162053,-0.9958737054294969>, <0.1831613272120411,0.5541186317434517,-0.021927940013526576>, 0.5 }
    cylinder {  m*<-2.908079589840338,-5.289429130322966,-1.8129734199254401>, <0.1831613272120411,0.5541186317434517,-0.021927940013526576>, 0.5}

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
    sphere { m*<0.1831613272120411,0.5541186317434517,-0.021927940013526576>, 1 }        
    sphere {  m*<0.4238964319537327,0.6828287099237772,2.965626831107024>, 1 }
    sphere {  m*<2.9178697212182985,0.656152607129826,-1.2511374654647112>, 1 }
    sphere {  m*<-1.4384540326808493,2.882592576162053,-0.9958737054294969>, 1}
    sphere { m*<-2.908079589840338,-5.289429130322966,-1.8129734199254401>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4238964319537327,0.6828287099237772,2.965626831107024>, <0.1831613272120411,0.5541186317434517,-0.021927940013526576>, 0.5 }
    cylinder { m*<2.9178697212182985,0.656152607129826,-1.2511374654647112>, <0.1831613272120411,0.5541186317434517,-0.021927940013526576>, 0.5}
    cylinder { m*<-1.4384540326808493,2.882592576162053,-0.9958737054294969>, <0.1831613272120411,0.5541186317434517,-0.021927940013526576>, 0.5 }
    cylinder {  m*<-2.908079589840338,-5.289429130322966,-1.8129734199254401>, <0.1831613272120411,0.5541186317434517,-0.021927940013526576>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    