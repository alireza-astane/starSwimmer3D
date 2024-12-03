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
    sphere { m*<-0.26257521012839047,-0.13560478529665657,-1.6142942771676743>, 1 }        
    sphere {  m*<0.519653390212279,0.2826171620706133,8.093267025793365>, 1 }
    sphere {  m*<2.4721331838778666,-0.03357080991028247,-2.8435038026188577>, 1 }
    sphere {  m*<-1.8841905700212807,2.1928691591219422,-2.588240042583644>, 1}
    sphere { m*<-1.6164033489834488,-2.694822783281955,-2.3986937574210714>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.519653390212279,0.2826171620706133,8.093267025793365>, <-0.26257521012839047,-0.13560478529665657,-1.6142942771676743>, 0.5 }
    cylinder { m*<2.4721331838778666,-0.03357080991028247,-2.8435038026188577>, <-0.26257521012839047,-0.13560478529665657,-1.6142942771676743>, 0.5}
    cylinder { m*<-1.8841905700212807,2.1928691591219422,-2.588240042583644>, <-0.26257521012839047,-0.13560478529665657,-1.6142942771676743>, 0.5 }
    cylinder {  m*<-1.6164033489834488,-2.694822783281955,-2.3986937574210714>, <-0.26257521012839047,-0.13560478529665657,-1.6142942771676743>, 0.5}

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
    sphere { m*<-0.26257521012839047,-0.13560478529665657,-1.6142942771676743>, 1 }        
    sphere {  m*<0.519653390212279,0.2826171620706133,8.093267025793365>, 1 }
    sphere {  m*<2.4721331838778666,-0.03357080991028247,-2.8435038026188577>, 1 }
    sphere {  m*<-1.8841905700212807,2.1928691591219422,-2.588240042583644>, 1}
    sphere { m*<-1.6164033489834488,-2.694822783281955,-2.3986937574210714>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.519653390212279,0.2826171620706133,8.093267025793365>, <-0.26257521012839047,-0.13560478529665657,-1.6142942771676743>, 0.5 }
    cylinder { m*<2.4721331838778666,-0.03357080991028247,-2.8435038026188577>, <-0.26257521012839047,-0.13560478529665657,-1.6142942771676743>, 0.5}
    cylinder { m*<-1.8841905700212807,2.1928691591219422,-2.588240042583644>, <-0.26257521012839047,-0.13560478529665657,-1.6142942771676743>, 0.5 }
    cylinder {  m*<-1.6164033489834488,-2.694822783281955,-2.3986937574210714>, <-0.26257521012839047,-0.13560478529665657,-1.6142942771676743>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    