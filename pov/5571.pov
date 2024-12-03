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
    sphere { m*<-0.9956815380424129,-0.16545460159693243,-1.3493402680480624>, 1 }        
    sphere {  m*<0.206061324347103,0.28382527567325394,8.567990894330988>, 1 }
    sphere {  m*<5.269316575911958,0.05640044578991937,-4.472967292944497>, 1 }
    sphere {  m*<-2.6515308552990895,2.163491377931864,-2.2626666813854124>, 1}
    sphere { m*<-2.383743634261258,-2.7242005644720337,-2.0731203962228415>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.206061324347103,0.28382527567325394,8.567990894330988>, <-0.9956815380424129,-0.16545460159693243,-1.3493402680480624>, 0.5 }
    cylinder { m*<5.269316575911958,0.05640044578991937,-4.472967292944497>, <-0.9956815380424129,-0.16545460159693243,-1.3493402680480624>, 0.5}
    cylinder { m*<-2.6515308552990895,2.163491377931864,-2.2626666813854124>, <-0.9956815380424129,-0.16545460159693243,-1.3493402680480624>, 0.5 }
    cylinder {  m*<-2.383743634261258,-2.7242005644720337,-2.0731203962228415>, <-0.9956815380424129,-0.16545460159693243,-1.3493402680480624>, 0.5}

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
    sphere { m*<-0.9956815380424129,-0.16545460159693243,-1.3493402680480624>, 1 }        
    sphere {  m*<0.206061324347103,0.28382527567325394,8.567990894330988>, 1 }
    sphere {  m*<5.269316575911958,0.05640044578991937,-4.472967292944497>, 1 }
    sphere {  m*<-2.6515308552990895,2.163491377931864,-2.2626666813854124>, 1}
    sphere { m*<-2.383743634261258,-2.7242005644720337,-2.0731203962228415>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.206061324347103,0.28382527567325394,8.567990894330988>, <-0.9956815380424129,-0.16545460159693243,-1.3493402680480624>, 0.5 }
    cylinder { m*<5.269316575911958,0.05640044578991937,-4.472967292944497>, <-0.9956815380424129,-0.16545460159693243,-1.3493402680480624>, 0.5}
    cylinder { m*<-2.6515308552990895,2.163491377931864,-2.2626666813854124>, <-0.9956815380424129,-0.16545460159693243,-1.3493402680480624>, 0.5 }
    cylinder {  m*<-2.383743634261258,-2.7242005644720337,-2.0731203962228415>, <-0.9956815380424129,-0.16545460159693243,-1.3493402680480624>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    