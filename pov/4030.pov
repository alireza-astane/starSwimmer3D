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
    sphere { m*<-0.152489556379701,-0.07674700833999092,-0.2481166254217134>, 1 }        
    sphere {  m*<0.10565872093830614,0.0612730988287111,2.9555379408098896>, 1 }
    sphere {  m*<2.582218837626556,0.025286967046383196,-1.477326150872897>, 1 }
    sphere {  m*<-1.774104916272591,2.251726936078608,-1.2220623908376835>, 1}
    sphere { m*<-1.5063176952347592,-2.6359650063252893,-1.0325161056751109>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10565872093830614,0.0612730988287111,2.9555379408098896>, <-0.152489556379701,-0.07674700833999092,-0.2481166254217134>, 0.5 }
    cylinder { m*<2.582218837626556,0.025286967046383196,-1.477326150872897>, <-0.152489556379701,-0.07674700833999092,-0.2481166254217134>, 0.5}
    cylinder { m*<-1.774104916272591,2.251726936078608,-1.2220623908376835>, <-0.152489556379701,-0.07674700833999092,-0.2481166254217134>, 0.5 }
    cylinder {  m*<-1.5063176952347592,-2.6359650063252893,-1.0325161056751109>, <-0.152489556379701,-0.07674700833999092,-0.2481166254217134>, 0.5}

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
    sphere { m*<-0.152489556379701,-0.07674700833999092,-0.2481166254217134>, 1 }        
    sphere {  m*<0.10565872093830614,0.0612730988287111,2.9555379408098896>, 1 }
    sphere {  m*<2.582218837626556,0.025286967046383196,-1.477326150872897>, 1 }
    sphere {  m*<-1.774104916272591,2.251726936078608,-1.2220623908376835>, 1}
    sphere { m*<-1.5063176952347592,-2.6359650063252893,-1.0325161056751109>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10565872093830614,0.0612730988287111,2.9555379408098896>, <-0.152489556379701,-0.07674700833999092,-0.2481166254217134>, 0.5 }
    cylinder { m*<2.582218837626556,0.025286967046383196,-1.477326150872897>, <-0.152489556379701,-0.07674700833999092,-0.2481166254217134>, 0.5}
    cylinder { m*<-1.774104916272591,2.251726936078608,-1.2220623908376835>, <-0.152489556379701,-0.07674700833999092,-0.2481166254217134>, 0.5 }
    cylinder {  m*<-1.5063176952347592,-2.6359650063252893,-1.0325161056751109>, <-0.152489556379701,-0.07674700833999092,-0.2481166254217134>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    