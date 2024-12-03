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
    sphere { m*<-0.21219221090600515,-0.10866729408269069,-0.9890345342161426>, 1 }        
    sphere {  m*<0.3520908746085195,0.19302912917199144,6.013793849290424>, 1 }
    sphere {  m*<2.522516183100252,-0.006633318696316462,-2.218244059667323>, 1 }
    sphere {  m*<-1.833807570798895,2.219806650335908,-1.9629802996321102>, 1}
    sphere { m*<-1.5660203497610632,-2.6678852920679894,-1.7734340144695373>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3520908746085195,0.19302912917199144,6.013793849290424>, <-0.21219221090600515,-0.10866729408269069,-0.9890345342161426>, 0.5 }
    cylinder { m*<2.522516183100252,-0.006633318696316462,-2.218244059667323>, <-0.21219221090600515,-0.10866729408269069,-0.9890345342161426>, 0.5}
    cylinder { m*<-1.833807570798895,2.219806650335908,-1.9629802996321102>, <-0.21219221090600515,-0.10866729408269069,-0.9890345342161426>, 0.5 }
    cylinder {  m*<-1.5660203497610632,-2.6678852920679894,-1.7734340144695373>, <-0.21219221090600515,-0.10866729408269069,-0.9890345342161426>, 0.5}

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
    sphere { m*<-0.21219221090600515,-0.10866729408269069,-0.9890345342161426>, 1 }        
    sphere {  m*<0.3520908746085195,0.19302912917199144,6.013793849290424>, 1 }
    sphere {  m*<2.522516183100252,-0.006633318696316462,-2.218244059667323>, 1 }
    sphere {  m*<-1.833807570798895,2.219806650335908,-1.9629802996321102>, 1}
    sphere { m*<-1.5660203497610632,-2.6678852920679894,-1.7734340144695373>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3520908746085195,0.19302912917199144,6.013793849290424>, <-0.21219221090600515,-0.10866729408269069,-0.9890345342161426>, 0.5 }
    cylinder { m*<2.522516183100252,-0.006633318696316462,-2.218244059667323>, <-0.21219221090600515,-0.10866729408269069,-0.9890345342161426>, 0.5}
    cylinder { m*<-1.833807570798895,2.219806650335908,-1.9629802996321102>, <-0.21219221090600515,-0.10866729408269069,-0.9890345342161426>, 0.5 }
    cylinder {  m*<-1.5660203497610632,-2.6678852920679894,-1.7734340144695373>, <-0.21219221090600515,-0.10866729408269069,-0.9890345342161426>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    