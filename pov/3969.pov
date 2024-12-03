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
    sphere { m*<-0.13520953139604286,-0.04771583158281856,-0.20639000639604815>, 1 }        
    sphere {  m*<0.10552557334564866,0.08099424659750665,2.781164764724502>, 1 }
    sphere {  m*<2.5994988626102185,0.05431814380355582,-1.4355995318472332>, 1 }
    sphere {  m*<-1.756824891288936,2.280758112835784,-1.180335771812019>, 1}
    sphere { m*<-1.5838056399709455,-2.786079089427072,-1.045697450267173>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10552557334564866,0.08099424659750665,2.781164764724502>, <-0.13520953139604286,-0.04771583158281856,-0.20639000639604815>, 0.5 }
    cylinder { m*<2.5994988626102185,0.05431814380355582,-1.4355995318472332>, <-0.13520953139604286,-0.04771583158281856,-0.20639000639604815>, 0.5}
    cylinder { m*<-1.756824891288936,2.280758112835784,-1.180335771812019>, <-0.13520953139604286,-0.04771583158281856,-0.20639000639604815>, 0.5 }
    cylinder {  m*<-1.5838056399709455,-2.786079089427072,-1.045697450267173>, <-0.13520953139604286,-0.04771583158281856,-0.20639000639604815>, 0.5}

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
    sphere { m*<-0.13520953139604286,-0.04771583158281856,-0.20639000639604815>, 1 }        
    sphere {  m*<0.10552557334564866,0.08099424659750665,2.781164764724502>, 1 }
    sphere {  m*<2.5994988626102185,0.05431814380355582,-1.4355995318472332>, 1 }
    sphere {  m*<-1.756824891288936,2.280758112835784,-1.180335771812019>, 1}
    sphere { m*<-1.5838056399709455,-2.786079089427072,-1.045697450267173>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10552557334564866,0.08099424659750665,2.781164764724502>, <-0.13520953139604286,-0.04771583158281856,-0.20639000639604815>, 0.5 }
    cylinder { m*<2.5994988626102185,0.05431814380355582,-1.4355995318472332>, <-0.13520953139604286,-0.04771583158281856,-0.20639000639604815>, 0.5}
    cylinder { m*<-1.756824891288936,2.280758112835784,-1.180335771812019>, <-0.13520953139604286,-0.04771583158281856,-0.20639000639604815>, 0.5 }
    cylinder {  m*<-1.5838056399709455,-2.786079089427072,-1.045697450267173>, <-0.13520953139604286,-0.04771583158281856,-0.20639000639604815>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    