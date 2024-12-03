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
    sphere { m*<-0.9655947487044465,-0.16439443374248935,-1.3644770723778508>, 1 }        
    sphere {  m*<0.22123222693082273,0.2841404341417015,8.554684926101631>, 1 }
    sphere {  m*<5.166213189910239,0.05318828753759755,-4.4102125925172935>, 1 }
    sphere {  m*<-2.6202065407911146,2.1645305259170176,-2.2800971130468906>, 1}
    sphere { m*<-2.3524193197532832,-2.72316141648688,-2.09055082788432>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.22123222693082273,0.2841404341417015,8.554684926101631>, <-0.9655947487044465,-0.16439443374248935,-1.3644770723778508>, 0.5 }
    cylinder { m*<5.166213189910239,0.05318828753759755,-4.4102125925172935>, <-0.9655947487044465,-0.16439443374248935,-1.3644770723778508>, 0.5}
    cylinder { m*<-2.6202065407911146,2.1645305259170176,-2.2800971130468906>, <-0.9655947487044465,-0.16439443374248935,-1.3644770723778508>, 0.5 }
    cylinder {  m*<-2.3524193197532832,-2.72316141648688,-2.09055082788432>, <-0.9655947487044465,-0.16439443374248935,-1.3644770723778508>, 0.5}

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
    sphere { m*<-0.9655947487044465,-0.16439443374248935,-1.3644770723778508>, 1 }        
    sphere {  m*<0.22123222693082273,0.2841404341417015,8.554684926101631>, 1 }
    sphere {  m*<5.166213189910239,0.05318828753759755,-4.4102125925172935>, 1 }
    sphere {  m*<-2.6202065407911146,2.1645305259170176,-2.2800971130468906>, 1}
    sphere { m*<-2.3524193197532832,-2.72316141648688,-2.09055082788432>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.22123222693082273,0.2841404341417015,8.554684926101631>, <-0.9655947487044465,-0.16439443374248935,-1.3644770723778508>, 0.5 }
    cylinder { m*<5.166213189910239,0.05318828753759755,-4.4102125925172935>, <-0.9655947487044465,-0.16439443374248935,-1.3644770723778508>, 0.5}
    cylinder { m*<-2.6202065407911146,2.1645305259170176,-2.2800971130468906>, <-0.9655947487044465,-0.16439443374248935,-1.3644770723778508>, 0.5 }
    cylinder {  m*<-2.3524193197532832,-2.72316141648688,-2.09055082788432>, <-0.9655947487044465,-0.16439443374248935,-1.3644770723778508>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    