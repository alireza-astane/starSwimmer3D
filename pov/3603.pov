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
    sphere { m*<0.07929368657479807,0.3577717486361859,-0.0821081955368771>, 1 }        
    sphere {  m*<0.32002879131648965,0.4864818268165112,2.9054465755836727>, 1 }
    sphere {  m*<2.814002080581057,0.45980572402256015,-1.3113177209880629>, 1 }
    sphere {  m*<-1.5423216733180927,2.6862456930547873,-1.0560539609528483>, 1}
    sphere { m*<-2.5254716525823286,-4.566163679487514,-1.591292795544403>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.32002879131648965,0.4864818268165112,2.9054465755836727>, <0.07929368657479807,0.3577717486361859,-0.0821081955368771>, 0.5 }
    cylinder { m*<2.814002080581057,0.45980572402256015,-1.3113177209880629>, <0.07929368657479807,0.3577717486361859,-0.0821081955368771>, 0.5}
    cylinder { m*<-1.5423216733180927,2.6862456930547873,-1.0560539609528483>, <0.07929368657479807,0.3577717486361859,-0.0821081955368771>, 0.5 }
    cylinder {  m*<-2.5254716525823286,-4.566163679487514,-1.591292795544403>, <0.07929368657479807,0.3577717486361859,-0.0821081955368771>, 0.5}

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
    sphere { m*<0.07929368657479807,0.3577717486361859,-0.0821081955368771>, 1 }        
    sphere {  m*<0.32002879131648965,0.4864818268165112,2.9054465755836727>, 1 }
    sphere {  m*<2.814002080581057,0.45980572402256015,-1.3113177209880629>, 1 }
    sphere {  m*<-1.5423216733180927,2.6862456930547873,-1.0560539609528483>, 1}
    sphere { m*<-2.5254716525823286,-4.566163679487514,-1.591292795544403>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.32002879131648965,0.4864818268165112,2.9054465755836727>, <0.07929368657479807,0.3577717486361859,-0.0821081955368771>, 0.5 }
    cylinder { m*<2.814002080581057,0.45980572402256015,-1.3113177209880629>, <0.07929368657479807,0.3577717486361859,-0.0821081955368771>, 0.5}
    cylinder { m*<-1.5423216733180927,2.6862456930547873,-1.0560539609528483>, <0.07929368657479807,0.3577717486361859,-0.0821081955368771>, 0.5 }
    cylinder {  m*<-2.5254716525823286,-4.566163679487514,-1.591292795544403>, <0.07929368657479807,0.3577717486361859,-0.0821081955368771>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    