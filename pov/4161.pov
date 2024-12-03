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
    sphere { m*<-0.16493425957657118,-0.08340062341253628,-0.4025570525911617>, 1 }        
    sphere {  m*<0.16679871475747954,0.09396186431995177,3.714293422012955>, 1 }
    sphere {  m*<2.569774134429686,0.01863335197383785,-1.6317665780423452>, 1 }
    sphere {  m*<-1.7865496194694612,2.2450733210060623,-1.3765028180071317>, 1}
    sphere { m*<-1.5187623984316294,-2.642618621397835,-1.186956532844559>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.16679871475747954,0.09396186431995177,3.714293422012955>, <-0.16493425957657118,-0.08340062341253628,-0.4025570525911617>, 0.5 }
    cylinder { m*<2.569774134429686,0.01863335197383785,-1.6317665780423452>, <-0.16493425957657118,-0.08340062341253628,-0.4025570525911617>, 0.5}
    cylinder { m*<-1.7865496194694612,2.2450733210060623,-1.3765028180071317>, <-0.16493425957657118,-0.08340062341253628,-0.4025570525911617>, 0.5 }
    cylinder {  m*<-1.5187623984316294,-2.642618621397835,-1.186956532844559>, <-0.16493425957657118,-0.08340062341253628,-0.4025570525911617>, 0.5}

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
    sphere { m*<-0.16493425957657118,-0.08340062341253628,-0.4025570525911617>, 1 }        
    sphere {  m*<0.16679871475747954,0.09396186431995177,3.714293422012955>, 1 }
    sphere {  m*<2.569774134429686,0.01863335197383785,-1.6317665780423452>, 1 }
    sphere {  m*<-1.7865496194694612,2.2450733210060623,-1.3765028180071317>, 1}
    sphere { m*<-1.5187623984316294,-2.642618621397835,-1.186956532844559>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.16679871475747954,0.09396186431995177,3.714293422012955>, <-0.16493425957657118,-0.08340062341253628,-0.4025570525911617>, 0.5 }
    cylinder { m*<2.569774134429686,0.01863335197383785,-1.6317665780423452>, <-0.16493425957657118,-0.08340062341253628,-0.4025570525911617>, 0.5}
    cylinder { m*<-1.7865496194694612,2.2450733210060623,-1.3765028180071317>, <-0.16493425957657118,-0.08340062341253628,-0.4025570525911617>, 0.5 }
    cylinder {  m*<-1.5187623984316294,-2.642618621397835,-1.186956532844559>, <-0.16493425957657118,-0.08340062341253628,-0.4025570525911617>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    