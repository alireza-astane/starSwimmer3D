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
    sphere { m*<-0.252274248176206,-0.13009733087026187,-1.486457964884091>, 1 }        
    sphere {  m*<0.4867022783458722,0.2649997057839426,7.6843393308656145>, 1 }
    sphere {  m*<2.482434145830051,-0.028063355483887706,-2.7156674903352727>, 1 }
    sphere {  m*<-1.8738896080690959,2.198376613548337,-2.4604037303000594>, 1}
    sphere { m*<-1.606102387031264,-2.6893153288555602,-2.2708574451374863>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4867022783458722,0.2649997057839426,7.6843393308656145>, <-0.252274248176206,-0.13009733087026187,-1.486457964884091>, 0.5 }
    cylinder { m*<2.482434145830051,-0.028063355483887706,-2.7156674903352727>, <-0.252274248176206,-0.13009733087026187,-1.486457964884091>, 0.5}
    cylinder { m*<-1.8738896080690959,2.198376613548337,-2.4604037303000594>, <-0.252274248176206,-0.13009733087026187,-1.486457964884091>, 0.5 }
    cylinder {  m*<-1.606102387031264,-2.6893153288555602,-2.2708574451374863>, <-0.252274248176206,-0.13009733087026187,-1.486457964884091>, 0.5}

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
    sphere { m*<-0.252274248176206,-0.13009733087026187,-1.486457964884091>, 1 }        
    sphere {  m*<0.4867022783458722,0.2649997057839426,7.6843393308656145>, 1 }
    sphere {  m*<2.482434145830051,-0.028063355483887706,-2.7156674903352727>, 1 }
    sphere {  m*<-1.8738896080690959,2.198376613548337,-2.4604037303000594>, 1}
    sphere { m*<-1.606102387031264,-2.6893153288555602,-2.2708574451374863>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4867022783458722,0.2649997057839426,7.6843393308656145>, <-0.252274248176206,-0.13009733087026187,-1.486457964884091>, 0.5 }
    cylinder { m*<2.482434145830051,-0.028063355483887706,-2.7156674903352727>, <-0.252274248176206,-0.13009733087026187,-1.486457964884091>, 0.5}
    cylinder { m*<-1.8738896080690959,2.198376613548337,-2.4604037303000594>, <-0.252274248176206,-0.13009733087026187,-1.486457964884091>, 0.5 }
    cylinder {  m*<-1.606102387031264,-2.6893153288555602,-2.2708574451374863>, <-0.252274248176206,-0.13009733087026187,-1.486457964884091>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    