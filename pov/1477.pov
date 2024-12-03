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
    sphere { m*<0.6524355701684261,-6.136776576987162e-18,0.943800296896111>, 1 }        
    sphere {  m*<0.7524791794858556,-2.2397447269694376e-18,3.942135235408829>, 1 }
    sphere {  m*<6.848373509384849,2.316784104963112e-18,-1.4823545151587127>, 1 }
    sphere {  m*<-4.170152811941815,8.164965809277259,-2.2304796024230624>, 1}
    sphere { m*<-4.170152811941815,-8.164965809277259,-2.230479602423066>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7524791794858556,-2.2397447269694376e-18,3.942135235408829>, <0.6524355701684261,-6.136776576987162e-18,0.943800296896111>, 0.5 }
    cylinder { m*<6.848373509384849,2.316784104963112e-18,-1.4823545151587127>, <0.6524355701684261,-6.136776576987162e-18,0.943800296896111>, 0.5}
    cylinder { m*<-4.170152811941815,8.164965809277259,-2.2304796024230624>, <0.6524355701684261,-6.136776576987162e-18,0.943800296896111>, 0.5 }
    cylinder {  m*<-4.170152811941815,-8.164965809277259,-2.230479602423066>, <0.6524355701684261,-6.136776576987162e-18,0.943800296896111>, 0.5}

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
    sphere { m*<0.6524355701684261,-6.136776576987162e-18,0.943800296896111>, 1 }        
    sphere {  m*<0.7524791794858556,-2.2397447269694376e-18,3.942135235408829>, 1 }
    sphere {  m*<6.848373509384849,2.316784104963112e-18,-1.4823545151587127>, 1 }
    sphere {  m*<-4.170152811941815,8.164965809277259,-2.2304796024230624>, 1}
    sphere { m*<-4.170152811941815,-8.164965809277259,-2.230479602423066>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7524791794858556,-2.2397447269694376e-18,3.942135235408829>, <0.6524355701684261,-6.136776576987162e-18,0.943800296896111>, 0.5 }
    cylinder { m*<6.848373509384849,2.316784104963112e-18,-1.4823545151587127>, <0.6524355701684261,-6.136776576987162e-18,0.943800296896111>, 0.5}
    cylinder { m*<-4.170152811941815,8.164965809277259,-2.2304796024230624>, <0.6524355701684261,-6.136776576987162e-18,0.943800296896111>, 0.5 }
    cylinder {  m*<-4.170152811941815,-8.164965809277259,-2.230479602423066>, <0.6524355701684261,-6.136776576987162e-18,0.943800296896111>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    