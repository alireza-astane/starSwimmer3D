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
    sphere { m*<-0.5553604849538543,-0.14939261213048793,-1.5567397597003574>, 1 }        
    sphere {  m*<0.41349919849345745,0.2881550757217035,8.386583367941222>, 1 }
    sphere {  m*<3.684681034199717,0.005492934388482779,-3.548249620473343>, 1 }
    sphere {  m*<-2.191521780626404,2.1792552207329887,-2.505618300666381>, 1}
    sphere { m*<-1.923734559588572,-2.7084367216709087,-2.3160720155038104>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.41349919849345745,0.2881550757217035,8.386583367941222>, <-0.5553604849538543,-0.14939261213048793,-1.5567397597003574>, 0.5 }
    cylinder { m*<3.684681034199717,0.005492934388482779,-3.548249620473343>, <-0.5553604849538543,-0.14939261213048793,-1.5567397597003574>, 0.5}
    cylinder { m*<-2.191521780626404,2.1792552207329887,-2.505618300666381>, <-0.5553604849538543,-0.14939261213048793,-1.5567397597003574>, 0.5 }
    cylinder {  m*<-1.923734559588572,-2.7084367216709087,-2.3160720155038104>, <-0.5553604849538543,-0.14939261213048793,-1.5567397597003574>, 0.5}

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
    sphere { m*<-0.5553604849538543,-0.14939261213048793,-1.5567397597003574>, 1 }        
    sphere {  m*<0.41349919849345745,0.2881550757217035,8.386583367941222>, 1 }
    sphere {  m*<3.684681034199717,0.005492934388482779,-3.548249620473343>, 1 }
    sphere {  m*<-2.191521780626404,2.1792552207329887,-2.505618300666381>, 1}
    sphere { m*<-1.923734559588572,-2.7084367216709087,-2.3160720155038104>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.41349919849345745,0.2881550757217035,8.386583367941222>, <-0.5553604849538543,-0.14939261213048793,-1.5567397597003574>, 0.5 }
    cylinder { m*<3.684681034199717,0.005492934388482779,-3.548249620473343>, <-0.5553604849538543,-0.14939261213048793,-1.5567397597003574>, 0.5}
    cylinder { m*<-2.191521780626404,2.1792552207329887,-2.505618300666381>, <-0.5553604849538543,-0.14939261213048793,-1.5567397597003574>, 0.5 }
    cylinder {  m*<-1.923734559588572,-2.7084367216709087,-2.3160720155038104>, <-0.5553604849538543,-0.14939261213048793,-1.5567397597003574>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    