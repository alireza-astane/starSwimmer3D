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
    sphere { m*<-0.694421746745343,-1.0328552054534936,-0.5711420708083913>, 1 }        
    sphere {  m*<0.7247457474548195,-0.04291629157357568,9.278148026226761>, 1 }
    sphere {  m*<8.092532945777613,-0.3280085423658389,-5.292529402847172>, 1 }
    sphere {  m*<-6.803430247911369,6.195072831254813,-3.8017224996655683>, 1}
    sphere { m*<-2.634814788225188,-5.2586523728903956,-1.4697139292039136>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7247457474548195,-0.04291629157357568,9.278148026226761>, <-0.694421746745343,-1.0328552054534936,-0.5711420708083913>, 0.5 }
    cylinder { m*<8.092532945777613,-0.3280085423658389,-5.292529402847172>, <-0.694421746745343,-1.0328552054534936,-0.5711420708083913>, 0.5}
    cylinder { m*<-6.803430247911369,6.195072831254813,-3.8017224996655683>, <-0.694421746745343,-1.0328552054534936,-0.5711420708083913>, 0.5 }
    cylinder {  m*<-2.634814788225188,-5.2586523728903956,-1.4697139292039136>, <-0.694421746745343,-1.0328552054534936,-0.5711420708083913>, 0.5}

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
    sphere { m*<-0.694421746745343,-1.0328552054534936,-0.5711420708083913>, 1 }        
    sphere {  m*<0.7247457474548195,-0.04291629157357568,9.278148026226761>, 1 }
    sphere {  m*<8.092532945777613,-0.3280085423658389,-5.292529402847172>, 1 }
    sphere {  m*<-6.803430247911369,6.195072831254813,-3.8017224996655683>, 1}
    sphere { m*<-2.634814788225188,-5.2586523728903956,-1.4697139292039136>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7247457474548195,-0.04291629157357568,9.278148026226761>, <-0.694421746745343,-1.0328552054534936,-0.5711420708083913>, 0.5 }
    cylinder { m*<8.092532945777613,-0.3280085423658389,-5.292529402847172>, <-0.694421746745343,-1.0328552054534936,-0.5711420708083913>, 0.5}
    cylinder { m*<-6.803430247911369,6.195072831254813,-3.8017224996655683>, <-0.694421746745343,-1.0328552054534936,-0.5711420708083913>, 0.5 }
    cylinder {  m*<-2.634814788225188,-5.2586523728903956,-1.4697139292039136>, <-0.694421746745343,-1.0328552054534936,-0.5711420708083913>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    