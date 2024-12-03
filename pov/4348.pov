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
    sphere { m*<-0.18510584776593905,-0.09418545139041128,-0.652889153515486>, 1 }        
    sphere {  m*<0.251667877270403,0.13933753378809197,4.767531052887564>, 1 }
    sphere {  m*<2.549602546240318,0.007848523995962906,-1.882098678966668>, 1 }
    sphere {  m*<-1.806721207658829,2.2342884930281874,-1.6268349189314548>, 1}
    sphere { m*<-1.5389339866209972,-2.65340344937571,-1.437288633768882>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.251667877270403,0.13933753378809197,4.767531052887564>, <-0.18510584776593905,-0.09418545139041128,-0.652889153515486>, 0.5 }
    cylinder { m*<2.549602546240318,0.007848523995962906,-1.882098678966668>, <-0.18510584776593905,-0.09418545139041128,-0.652889153515486>, 0.5}
    cylinder { m*<-1.806721207658829,2.2342884930281874,-1.6268349189314548>, <-0.18510584776593905,-0.09418545139041128,-0.652889153515486>, 0.5 }
    cylinder {  m*<-1.5389339866209972,-2.65340344937571,-1.437288633768882>, <-0.18510584776593905,-0.09418545139041128,-0.652889153515486>, 0.5}

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
    sphere { m*<-0.18510584776593905,-0.09418545139041128,-0.652889153515486>, 1 }        
    sphere {  m*<0.251667877270403,0.13933753378809197,4.767531052887564>, 1 }
    sphere {  m*<2.549602546240318,0.007848523995962906,-1.882098678966668>, 1 }
    sphere {  m*<-1.806721207658829,2.2342884930281874,-1.6268349189314548>, 1}
    sphere { m*<-1.5389339866209972,-2.65340344937571,-1.437288633768882>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.251667877270403,0.13933753378809197,4.767531052887564>, <-0.18510584776593905,-0.09418545139041128,-0.652889153515486>, 0.5 }
    cylinder { m*<2.549602546240318,0.007848523995962906,-1.882098678966668>, <-0.18510584776593905,-0.09418545139041128,-0.652889153515486>, 0.5}
    cylinder { m*<-1.806721207658829,2.2342884930281874,-1.6268349189314548>, <-0.18510584776593905,-0.09418545139041128,-0.652889153515486>, 0.5 }
    cylinder {  m*<-1.5389339866209972,-2.65340344937571,-1.437288633768882>, <-0.18510584776593905,-0.09418545139041128,-0.652889153515486>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    